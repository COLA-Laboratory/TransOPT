import atexit
import json
import queue
import sqlite3
import time
from multiprocessing import Event, Manager, Process, Queue
from typing import Union

import numpy as np
import pandas as pd

from transopt.utils.log import logger
from transopt.utils.path import get_library_path

"""
Descriptions of the reserved database tables.
"""
table_descriptions = {
    "_config": """
        name varchar(200) not null,
        config text not null,
        is_experiment boolean not null default TRUE
        """,
    "_metadata": """
        table_name varchar(255) not null,
        problem_name varchar(255) not null,
        dimensions int,
        objectives int,
        fidelities text,
        workloads int,
        budget_type varchar(50),
        budget int,
        seeds int,
        space_refiner varchar(50),
        sampler varchar(50),
        pretrain varchar(50),
        model varchar(50),
        acf varchar(50),
        normalizer varchar(50),
        dataset_selectors json,
        PRIMARY KEY (table_name)
    """,
}


class DatabaseDaemon:
    def __init__(self, data_path, task_queue, result_queue, stop_event):
        self.data_path = data_path
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event

    def run(self):
        with sqlite3.connect(self.data_path) as conn:
            cursor = conn.cursor()
            while not self.stop_event.is_set():
                task = self.task_queue.get()  # Check every second
                if task is None:  # Sentinel for stopping
                    break
                func, args, commit = task
                try:
                    result = func(cursor, *args)
                    if commit:
                        conn.commit()
                    self.result_queue.put(("SUCCESS", result))
                except Exception as e:
                    conn.rollback()
                    logger.error(
                        f"Database operation failed: {e}", exc_info=True
                    )
                    self.result_queue.put(("FAILURE", e))


class Database:
    def __init__(self, db_file_name="database.db"):
        self.data_path = get_library_path() / db_file_name

        manager = Manager()
        self.task_queue = manager.Queue()
        self.result_queue = manager.Queue()
        self.lock = manager.Lock()
        self.transaction_lock = manager.Lock()
        self.stop_event = manager.Event()

        self.process = Process(
            target=DatabaseDaemon(
                self.data_path, self.task_queue, self.result_queue, self.stop_event
            ).run
        )
        self.process.start()
        atexit.register(self.close)

        # reserved tables
        self.reserved_tables = list(table_descriptions.keys())
        for name, desc in table_descriptions.items():
            if not self.check_table_exist(name):
                self.execute(f'CREATE TABLE "{name}" ({desc})')

    def close(self):
        self.stop_event.set()
        self.task_queue.put(None)
        self.process.join()

    def _execute(self, task, args=(), timeout=None, commit=True):
        self.task_queue.put((task, args, commit))
        try:
            status, result = self.result_queue.get(timeout=timeout)
            if status == "SUCCESS":
                return result
            else:
                raise result  # Re-raise the exception from the daemon
        except queue.Empty:
            raise Exception("Task execution timed out or failed")

    @staticmethod
    def query_exec(cursor, query, params, fetchone, fetchall, many):
        if many:
            cursor.executemany(query, params or [])
        else:
            cursor.execute(query, params or ())
        if fetchone:
            return cursor.fetchone()
        if fetchall:
            return cursor.fetchall()
        return None

    def execute(
        self,
        query,
        params=None,
        fetchone=False,
        fetchall=False,
        timeout=None,
        commit=True,
    ):
        with self.lock:
            return self._execute(
                Database.query_exec,
                (query, params, fetchone, fetchall, False),
                timeout,
                commit
            )

    def executemany(
        self,
        query,
        params=None,
        fetchone=False,
        fetchall=False,
        timeout=None,
        commit=True,
    ):
        with self.lock:
            return self._execute(
                Database.query_exec,
                (query, params, fetchone, fetchall, True),
                timeout,
                commit
            )

    def start_transaction(self):
        self.execute("BEGIN", commit=False)

    def commit_transaction(self):
        self.execute("COMMIT", commit=False)

    def rollback_transaction(self):
        self.execute("ROLLBACK", commit=False)
        
    """ 
    table
    """

    def get_experiment_datasets(self):
        """Get the list of all tables that are marked as experiment datasets."""
        experiment_datasets = self.execute(
            "SELECT name FROM _config WHERE is_experiment = TRUE", fetchall=True
        )
        return [
            table[0]
            for table in experiment_datasets
            if table[0] not in self.reserved_tables
        ]

    def get_all_datasets(self):
        """Get the list of all tables and indicate which ones are experiment datasets."""
        all_datasets = self.execute(
            """SELECT name, is_experiment FROM _config""", fetchall=True
        )
        return [
            table[0] for table in all_datasets if table[0] not in self.reserved_tables
        ]

    def get_table_list(self):
        """Get the list of all database tables."""
        table_list = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table'", fetchall=True
        )
        return [
            table[0] for table in table_list if table[0] not in self.reserved_tables
        ]

    def check_table_exist(self, name):
        """Check if a certain database table exists."""
        table_exists = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            params=(name,),
            fetchone=True,
        )
        return table_exists is not None

    def create_table(self, name, dataset_cfg, overwrite=False, is_experiment=True):
        """
        Create and initialize a database table based on problem configuration.

        Parameters
        ----------
        name: str
            Name of the table to create and initialize.
        dataset_cfg: dict
            Configuration for the table schema.
        overwrite : bool, optional
            Flag to determine whether to overwrite the existing table, default is False.
        is_experiment : bool, optional
            Flag to denote if the table is for experimental use, default is True.
        """
        if self.check_table_exist(name):
            if overwrite:
                self.remove_table(name)
            else:
                raise Exception(f"Table {name} already exists")

        variables = dataset_cfg.get("variables", [])
        objectives = dataset_cfg.get("objectives", [])
        fidelities = dataset_cfg.get("fidelities", [])

        var_type_map = {
            "continuous": "float",
            "log_continuous": "float",
            "integer": "int",
            "large_integer": "text",  # Store large integers as text to handle very large values
            "int_exponent": "int",
            "exp2": "int",
            "categorical": "varchar(50)",
            # 'binary': 'boolean',
        }

        # description = ['status varchar(20) not null default "unevaluated"']
        description = []

        for var_info in variables:
            description.append(
                f'"{var_info["name"]}" {var_type_map[var_info["type"]]} not null'
            )

        for obj_info in objectives:
            description.append(f'"{obj_info["name"]}" float')

        for fid_info in fidelities:
            description.append(
                f'"{fid_info["name"]}" {var_type_map[fid_info["type"]]} not null'
            )

        description += [
            "batch int default -1",
            "error boolean default 0",
            # "pareto boolean",
            # "batch int not null",
            # "order int default -1",
            # "hypervolume float",
        ]

        with self.transaction_lock:
            try:
                self.start_transaction()
            
                # Create the table
                self.execute(f'CREATE TABLE "{name}" ({",".join(description)})', commit=False)

                # Optionally, create indexes on certain columns
                index_columns = [var["name"] for var in variables] + [
                    fid["name"] for fid in fidelities if fid.get("index", False)
                ]
                if index_columns:
                    index_statement = ", ".join([f'"{col}"' for col in index_columns])
                    self.execute(f'CREATE INDEX "idx_{name}" ON "{name}" ({index_statement})', commit=False)

                self.create_or_update_config(name, dataset_cfg, is_experiment, commit=False)
                if "additional_config" in dataset_cfg:
                    self.create_or_update_metadata(name, dataset_cfg["additional_config"], commit=False)
            
                self.commit_transaction()
            except Exception as e:
                self.rollback_transaction()  # Rollback if an error occurred
                raise e

    def remove_table(self, name):
        if not self.check_table_exist(name):
            raise Exception(f"Table {name} does not exist")
        with self.transaction_lock:
            try:
                self.start_transaction()
                self.execute(f"DELETE FROM _config WHERE name = '{name}'", commit=False)
                self.execute(f"DELETE FROM _metadata WHERE table_name = '{name}'", commit=False)
                self.execute(f'DROP TABLE IF EXISTS "{name}"', commit=False)
                self.commit_transaction()
            except Exception as e:
                self.rollback_transaction()
                raise e

    """
    config
    """

    def create_or_update_config(self, name, dataset_cfg, is_experiment=True, commit=True):
        """
        Create or update a configuration entry in the _config table for a given table.
        """
        # Serialize dataset_cfg into JSON format
        config_json = json.dumps(dataset_cfg)

        # Check if the configuration already exists
        if self.query_config(name) is not None:
            # Update the existing configuration
            self.execute(
                "UPDATE _config SET config = ?, is_experiment = ? WHERE name = ?",
                (config_json, is_experiment, name),
                commit=commit
            )
        else:
            # Insert a new configuration
            self.execute(
                "INSERT INTO _config (name, config, is_experiment) VALUES (?, ?, ?)",
                (name, config_json, is_experiment),
                commit=commit
            )

    def query_config(self, name):
        config_json = self.execute(
            "SELECT config FROM _config WHERE name=?", params=(name,), fetchone=True
        )

        if config_json is None:
            return None
        else:
            return json.loads(config_json[0])

    def query_dataset_info(self, name):
        """
        Query the dataset information of a given table.
        """
        config = self.query_config(name)

        if config is None:
            return None

        variables = config["variables"]
        objectives = config["objectives"]
        fidelities = config["fidelities"]

        num_rows = self.get_num_row(name)

        dataset_info = {
            "num_variables": len(variables),
            "num_objectives": len(objectives),
            "num_fidelities": len(fidelities),
            "data_number": num_rows,
            **config,
        }
        return dataset_info

    def create_or_update_metadata(self, table_name, metadata, commit=True):
        """
        Create or update a metadata entry in the _metadata table for a given table.
        """
        dataset_selectors_json = json.dumps(metadata.get("DatasetSelectors", {}))
        problem_name = metadata.get("problem_name", "")

        self.execute(
            f"""
            INSERT INTO _metadata (
                table_name, problem_name, dimensions, objectives, fidelities, workloads, budget_type, budget, seeds,
                space_refiner, sampler, pretrain, model, acf, normalizer, dataset_selectors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (table_name) DO UPDATE SET
                problem_name = EXCLUDED.problem_name, dimensions = EXCLUDED.dimensions, objectives = EXCLUDED.objectives, 
                fidelities = EXCLUDED.fidelities, workloads = EXCLUDED.workloads, budget_type = EXCLUDED.budget_type,
                budget = EXCLUDED.budget, seeds = EXCLUDED.seeds, space_refiner = EXCLUDED.space_refiner,
                sampler = EXCLUDED.sampler, pretrain = EXCLUDED.pretrain, model = EXCLUDED.model, 
                acf = EXCLUDED.acf, normalizer = EXCLUDED.normalizer, dataset_selectors = EXCLUDED.dataset_selectors
            """,
            (
                table_name,
                problem_name,
                metadata.get("dim", 0),
                metadata.get("obj", 0),
                metadata.get("fidelity", ""),
                metadata.get("workloads", 0),
                metadata.get("budget_type", ""),
                metadata.get("budget", 0),
                metadata.get("seeds", 0),
                metadata.get("SpaceRefiner", ""),
                metadata.get("Sampler", ""),
                metadata.get("Pretrain", ""),
                metadata.get("Model", ""),
                metadata.get("ACF", ""),
                metadata.get("Normalizer", ""),
                dataset_selectors_json,
            ),
            commit=commit
        )

    def get_all_metadata(self):
        """
        Get the metadata for all tables in the database.
        """
        metadata = self.execute("SELECT * FROM _metadata", fetchall=True)
        return metadata

    def search_tables_by_metadata(self, search_params):
        """
        Search for tables based on metadata criteria.

        Parameters:
        ----------
        search_params : dict
            A dictionary where keys are metadata column names and values are the criteria values.

        Returns:
        -------
        list of str
            A list of table names that match the search criteria.
        """
        if not search_params:
            raise ValueError("Search parameters are required")

        # Constructing the WHERE clause dynamically based on the provided search parameters
        where_clause = self._get_conditions(conditions=search_params)

        query = f"SELECT table_name FROM _metadata{where_clause}"
        result = self.execute(query, fetchall=True)

        return [row[0] for row in result]

    """
    basic operations
    """

    def insert_data(
        self, table, data: Union[dict, list, pd.DataFrame, np.ndarray]
    ) -> list:
        """
        Insert single-row or multiple-row data into the database.

        Parameters
        ----------
        table: str
            Name of the database table to insert into.
        data: dict, list, pd.DataFrame, or np.ndarray
            Data to insert. If a dictionary, it represents a single row of data
            where keys are column names and values are data values. If a list,
            each element represents a row (as a list or dict). If a DataFrame or
            np.ndarray, each row represents a row to be inserted.

        Returns
        -------
        list
            List of row numbers of the inse

        """
        if isinstance(data, dict):
            # Single row insertion from dict
            columns = list(data.keys())
            values = [list(data.values())]
        elif isinstance(data, list):
            # Multiple row insertion from list of dicts or lists
            if all(isinstance(row, dict) for row in data):
                columns = list(data[0].keys())
                values = [list(row.values()) for row in data]
            elif all(isinstance(row, list) for row in data):
                columns = None
                values = data
            else:
                raise ValueError(
                    "All rows in data_list must be of the same type (all dicts or all lists)"
                )
        elif isinstance(data, (pd.DataFrame, np.ndarray)):
            # Convert DataFrame or ndarray to list of lists for insertion
            values = (
                data.tolist() if isinstance(data, np.ndarray) else data.values.tolist()
            )
            columns = data.columns.tolist() if isinstance(data, pd.DataFrame) else None
        else:
            raise ValueError(
                "Data parameter must be a dictionary, list, pandas DataFrame, or numpy ndarray"
            )

        if columns:
            column_str = ",".join([f'"{col}"' for col in columns])
            value_placeholders = ",".join(["?"] * len(columns))
        else:
            column_str = ""
            value_placeholders = ",".join(["?"] * len(values[0]))

        query = f'INSERT INTO "{table}" ({column_str}) VALUES ({value_placeholders})'
        self.executemany(query, values)

        # Get the rowids of the inserted rows
        n_row = self.get_num_row(table)
        len_data = len(data) if isinstance(data, list) else len(values)
        return list(range(n_row - len_data + 1, n_row + 1))

    def _get_conditions(self, rowid=None, conditions=None):
        """
        Construct SQL conditions for a query based on rowid and additional conditions.

        Parameters
        ----------
        rowid: int/list
            Row number(s) of the table to query (if None then no rowid condition is added).
        conditions: dict
            Additional conditions for querying (key: column name, value: column value).

        Returns
        -------
        str
            SQL condition string.
        """
        from collections.abc import Iterable

        conditions_list = []

        # Handling rowid conditions
        if rowid is not None:
            if isinstance(rowid, Iterable) and not isinstance(rowid, str):
                rowid_condition = f'rowid IN ({",".join([str(r) for r in rowid])})'
                conditions_list.append(rowid_condition)
            else:
                conditions_list.append(f"rowid = {rowid}")

        # Handling additional conditions
        if conditions:
            for column, value in conditions.items():
                if isinstance(value, str):
                    value_str = f"'{value}'"  # Strings need to be quoted
                else:
                    value_str = str(value)
                condition_str = f'"{column}" = {value_str}'
                conditions_list.append(condition_str)

        # Combine all conditions with 'AND'
        if conditions_list:
            return " WHERE " + " AND ".join(conditions_list)
        else:
            return ""

    def update_data(self, table, data, rowid=None, conditions=None):
        """
        Update single-row or multiple-row data in the database.

        Parameters
        ----------
        table: str
            Name of the database table to update.
        data: dict or list of dicts
            Data to update. If a dictionary, it represents a single row of data
            where keys are column names and values are data values.
            If a list, each dictionary in the list represents a row to be updated.
        rowid: int/list
            Row number(s) of the table to update. If None, conditions are used.
        conditions: dict
            Additional conditions for updating (key: column name, value: column value).
        """
        if isinstance(data, dict):
            data = [data]

        update_values = []
        for row in data:
            columns = list(row.keys())
            values = list(row.values())
            set_clause = ", ".join([f'"{col}" = ?' for col in columns])
            query = f'UPDATE "{table}" SET {set_clause}'

            if rowid:
                query += f" WHERE rowid = ?"
                values.append(rowid)
            elif conditions:
                condition_str = " AND ".join([f'"{k}" = ?' for k in conditions.keys()])
                query += f" WHERE {condition_str}"
                values.extend(conditions.values())
            else:
                raise ValueError("Either rowid or conditions must be provided")

            update_values.append(values)

        self.executemany(query, update_values)

    def delete_data(self, table, rowid=None, conditions=None):
        """
        Delete single-row or multiple-row data in the database.

        Parameters
        ----------
        table: str
            Name of the database table to delete from.
        rowid: int/list
            Row number(s) of the table to delete. If None, conditions are used.
        conditions: dict
            Additional conditions for deleting (key: column name, value: column value).
        """
        query = f'DELETE FROM "{table}"'
        condition = self._get_conditions(rowid=rowid, conditions=conditions)
        query += condition

        self.execute(query)

    def select_data(
        self, table, columns=None, rowid=None, conditions=None, as_dataframe=False
    ) -> Union[list, pd.DataFrame]:
        """
        Select data in the database.

        Parameters
        ----------
        table: str
            Name of the database table to query.
        column: str/list
            Column name(s) of the table to query (if None then select all columns).
        rowid: int/list
            Row number(s) of the table to query (if None then select all rows).
        conditions: dict
            Additional conditions for querying (key: column name, value: column value).
        as_dataframe: bool
            If True, return the result as a pandas DataFrame.

        Returns
        -------
        list of dicts
            Selected data, each row as a dictionary with column names as keys.
        """
        if columns is None:
            query = f'SELECT * FROM "{table}"'
            columns = self.get_column_names(table)
        elif isinstance(columns, str):
            query = f'SELECT "{columns}" FROM "{table}"'
            columns = [columns]
        else:
            column_str = ",".join([f'"{col}"' for col in columns])
            query = f'SELECT {column_str} FROM "{table}"'

        condition = self._get_conditions(rowid=rowid, conditions=conditions)
        query += condition

        # Convert each tuple in the results to a list
        results = self.execute(query, fetchall=True)
        if as_dataframe:
            return pd.DataFrame(results, columns=columns)
        else:
            # Convert large integer values from string to int if needed
            converted_results = []
            for row in results:
                row_dict = dict(zip(columns, row))
                for key, value in row_dict.items():
                    if isinstance(value, str) and value.isdigit() and key in self.large_integer_columns:
                        row_dict[key] = int(value)
                converted_results.append(row_dict)
            return converted_results

    def get_num_row(self, table):
        query = f'SELECT COUNT(*) FROM "{table}"'
        return self.execute(query, fetchone=True)[0]

    def get_column_names(self, table):
        """Get the column names of a database table."""
        query = f'PRAGMA table_info("{table}")'
        return [col[1] for col in self.execute(query, fetchall=True)]

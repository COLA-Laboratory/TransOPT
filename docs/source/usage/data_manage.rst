Data Management
===============

The `datamanager` module is designed to manage data generated during optimization tasks. It provides a structured approach for storing, querying, and transferring data across different optimization scenarios. This module is built with flexibility in mind, enabling efficient management of various optimization task requirements, from persistent storage to similarity-based searches.

The primary roles of the `datamanager` include:

- **Data Storage**: Utilizes a database-backed storage system to persist data generated throughout the optimization process, ensuring that configurations, results, and metadata are readily accessible.
- **Checkpointing**: Allows for saving the state of optimization tasks at specific points, facilitating recovery and continuation from those points in case of interruptions.
- **Flexible Search Mechanisms**: Incorporates both metadata-based searches for filtering and querying task information and Locality-Sensitive Hashing (LSH) for identifying similar data points within large datasets.

This combination of features makes the `datamanager` particularly valuable in scenarios where optimization processes are iterative and data-intensive, requiring a balance between precise control over individual task data and the ability to draw insights from historical records.

Data Storage and Management
---------------------------

The `datamanager` module's core lies in its robust data storage and management capabilities, allowing it to efficiently handle the various data generated during optimization tasks. At its heart, it utilizes an SQLite database to persistently store task-related data, configurations, and metadata, ensuring that important information is retained across task iterations.

Data Storage Mechanism
**********************
The `datamanager` uses SQLite as a backend for storing data, chosen for its lightweight and self-contained nature, making it suitable for scenarios where a full-fledged database system may be unnecessary. It supports various data formats for easy integration:

- **Dictionary**: Individual task data can be stored as dictionaries, with keys representing field names and values representing corresponding data.
- **List**: Supports lists of dictionaries or lists for batch insertion, allowing multiple records to be added in a single operation.
- **DataFrame/NumPy Arrays**: For more complex data structures, `datamanager` can insert rows from pandas DataFrames or NumPy arrays, automatically converting these into a database-compatible format.

This flexibility allows the `datamanager` to adapt to different data needs and store both simple and complex optimization data structures seamlessly.

Metadata Table Design
*********************
A key feature of the `datamanager` is its use of metadata tables to store descriptive information about each optimization task, enabling more efficient querying and organization of data. The `_metadata` table is specifically designed to record detailed information about each stored optimization task, such as:

- **table_name**: The name of the table where task data is stored, serving as the primary key.
- **problem_name**: The name of the optimization problem, providing context for the stored task.
- **dimensions**: The number of dimensions involved in the optimization problem.
- **objectives**: The number of objectives being optimized.
- **fidelities**: A textual representation of various fidelities or resolution levels within the task.
- **budget_type** and **budget**: Information about the type and limits of the budget used in the optimization.
- **space_refiner**, **sampler**, **model**, **pretrain**, **acf**, **normalizer**: Parameters related to the methodologies and models used in the optimization process.
- **dataset_selectors**: A JSON-encoded structure representing the criteria for dataset selection.

The `_metadata` table serves as a centralized index of all stored tasks, allowing users to quickly retrieve and filter tasks based on their descriptive attributes.

Data Storage Flexibility
************************
The design of the `datamanager` ensures that data storage remains flexible and adaptable. By supporting multiple input formats and allowing for dynamic database interactions, the module can handle diverse data requirements across different optimization tasks. This adaptability is crucial for optimization scenarios where the nature of the data may change based on the problem domain or the phase of the optimization process.

Through its use of a lightweight SQLite database, combined with a rich set of metadata, the `datamanager` provides a powerful yet simple way to manage and organize the data generated during optimization, laying the groundwork for more complex functionalities such as checkpointing and similarity-based searches.


Similarity Search and Data Reuse
--------------------------------

The `datamanager` module includes advanced functionality for identifying similar data points and facilitating data reuse between optimization tasks with similar characteristics. This is particularly valuable in scenarios where insights from past optimization runs can be leveraged to accelerate new tasks, reducing the time and computational effort needed to reach optimal solutions.

Similarity Search Mechanisms
****************************
The `datamanager` uses a combination of Locality-Sensitive Hashing (LSH) and MinHash techniques to perform similarity searches across stored optimization data. These techniques allow for efficient identification of data points that are similar but not identical, supporting exploratory optimization where near-optimal solutions can inform new tasks.

- **Locality-Sensitive Hashing (LSH)**: LSH is employed to map similar data points into the same hash buckets with high probability. This approach reduces the dimensionality of the data and enables fast similarity searches by grouping data that are likely to be similar into the same buckets.
- **MinHash Signatures**: MinHash is used to generate compact signatures for high-dimensional data (such as configuration states or text data). These signatures make it possible to estimate the similarity between data points by comparing their hashed values, which approximates the Jaccard similarity between sets.

Integration with the Manager for Task Similarity
************************************************
Within the `manager`, the LSH mechanism is used to identify optimization tasks that share similar characteristics. This process involves creating a concise representation, or "vector," of each task based on key properties like the number of variables, number of objectives, and descriptive information about the task.

- **Vector Representation of Tasks**: To enable comparison, each optimization task is represented by a vector that summarizes important aspects of the task, such as:
  - The complexity of the problem (e.g., number of variables and objectives).
  - Descriptive details about the variables, which can capture the nature of the problem.
  - The task's name or type, helping categorize similar tasks.

  This vector encapsulates the essential details of each optimization task, providing a structured way to describe the nature of the problem. By converting tasks into such standardized vectors, it becomes possible to use LSH to efficiently identify similar tasks.

- **Similarity Search Process**: The LSH mechanism uses these vectors to map tasks into clusters or hash buckets, where tasks with similar vectors are more likely to be grouped together. When a new task or query vector is introduced, the `manager` can quickly identify past tasks that fall into the same bucket, indicating similarity based on the problem structure and configuration.

  This allows the `datamanager` to efficiently locate tasks with similar characteristics, enabling reuse of past results or configurations that may be relevant to the new task.

Applications of Similarity-Based Data Reuse
*******************************************
The similarity search and data reuse capabilities of the `datamanager` provide significant advantages in various optimization scenarios:

- **Warm Start for Optimization**: By starting a new task with configurations similar to those that were effective in past tasks, users can perform "warm starts" that speed up convergence.
- **Adaptive Optimization**: For tasks that require adjusting optimization parameters over time, the ability to find and utilize past similar configurations ensures that the adjustments are more efficient.
- **Transfer Learning in Optimization**: When optimization tasks vary slightly across iterations (e.g., changing problem parameters or objectives), the `datamanager` helps carry over useful information from previous runs, acting as a form of transfer learning in optimization.

The integration of vector-based representations with LSH makes the `datamanager` a powerful tool for scenarios where similarity and reuse are critical. It enables users to not only store and manage data but also to leverage historical results for more efficient optimization processes.


Flexible Data Querying
----------------------

The `datamanager` module is equipped with versatile data querying capabilities, allowing users to perform both precise and approximate searches based on the needs of their optimization tasks. This flexibility ensures that data can be accessed efficiently, whether the goal is to find an exact match or to identify records that satisfy broader criteria.

Metadata-Based Search
*********************
A key feature of the `datamanager` is its ability to perform metadata-based searches. This type of search leverages the detailed metadata stored about each optimization task, enabling users to filter and retrieve data based on descriptive attributes such as task name, problem type, and configuration settings.

- **Dynamic Query Generation**: The `datamanager` constructs SQL `WHERE` clauses dynamically based on user-provided search criteria, allowing for flexible and complex queries. Users can specify one or multiple metadata fields as search parameters, and the module generates the appropriate SQL query to retrieve matching records.
- **Example**: Using the `search_tables_by_metadata` method, users can search for optimization tasks based on criteria such as `problem_name`, `budget_type`, or other attributes stored in the `_metadata` table:
  
  .. code-block:: python
  
      search_params = {
          "problem_name": "example_problem",
          "budget_type": "fixed"
      }
      matching_tables = datamanager.search_tables_by_metadata(search_params)

  This example retrieves the names of all tables associated with optimization tasks that match the given `problem_name` and `budget_type`, making it easier to filter data relevant to specific problem settings.

Precise Search and Filtering
****************************
While metadata-based search provides flexibility, the `datamanager` also supports precise search and filtering operations. These are particularly useful when specific task data needs to be retrieved without approximation:

- **Primary Key and Index-Based Search**: For tasks or configurations that are uniquely identified by fields like `table_name` or `task_id`, the `datamanager` uses indexed fields for fast lookups, ensuring that searches for individual records are highly efficient.
- **SQL Query Support**: Users can execute custom SQL queries to interact directly with the database, giving them complete control over the data retrieval process. This is useful when complex joins, aggregations, or advanced filtering conditions are required.
- **Example**: A precise search can be performed to retrieve data entries associated with a specific task ID:

  .. code-block:: python
  
      query = "SELECT * FROM _metadata WHERE table_name = 'task_example'"
      task_data = datamanager.execute(query, fetchall=True)

  This example demonstrates how to execute a custom SQL query to retrieve all metadata associated with a specific task table.

Combining Flexible and Precise Queries
**************************************
The true strength of the `datamanager` lies in its ability to combine flexible metadata-based search with precise data retrieval. This enables users to start with broad criteria to identify relevant tasks and then drill down into specific records for deeper analysis.

- **Hybrid Search Strategies**: Users can first use metadata-based queries to identify relevant tasks and then apply precise searches to extract detailed data from those tasks.
- **Example Workflow**:
  
  1. Use `search_tables_by_metadata` to find all tasks that match certain criteria (e.g., problem type).
  2. Iterate over the results and use SQL queries to retrieve detailed data from each identified task.

  .. code-block:: python
  
      search_params = {"problem_name": "example_problem"}
      tables = datamanager.search_tables_by_metadata(search_params)
      for table in tables:
          data = datamanager.execute(f"SELECT * FROM {table} WHERE objectives = 2", fetchall=True)

  This workflow allows users to identify relevant optimization tasks and then extract specific entries from each, providing a balance between breadth and depth in data retrieval.

The flexibility of the `datamanager`'s querying capabilities ensures that users can adapt their data retrieval strategies to their specific needs, whether they require a broad overview of multiple tasks or a detailed analysis of a single task. This makes it an indispensable tool for managing data in complex optimization environments.


Integration with Optimization Tasks
-----------------------------------

The `datamanager` module is designed to integrate smoothly with optimization task workflows, providing an interface for storing, retrieving, and managing data throughout the lifecycle of these tasks. It acts as the central repository for task configurations, results, and intermediate data, enabling easy access and modification during different stages of optimization.

Task Data Flow Management
*************************
The `datamanager` plays a crucial role in managing the flow of data during the execution of optimization tasks. It ensures that data is stored in a structured manner, facilitating efficient access and modification throughout the optimization process. Key aspects include:

- **Storing Initial Configurations**: When an optimization task starts, the initial configurations and parameters can be stored in the `datamanager`, creating a record of the starting point.
- **Recording Intermediate Results**: As the optimization progresses, intermediate results and states can be stored, allowing users to analyze the trajectory of the process.
- **Saving Final Outcomes**: Once the optimization task concludes, the final configurations and results are saved, creating a comprehensive record of the optimization process.

This structured approach to data storage ensures that all phases of the optimization process are properly recorded, making it easy to track changes and analyze outcomes.

Interaction with Optimization Algorithms
****************************************
The `datamanager` acts as a data backend that stores and retrieves task-related data as needed, allowing optimization algorithms to make data-driven decisions:

- **Configuration Retrieval**: Optimization algorithms can query the `datamanager` to retrieve specific configurations or parameters that were effective in past tasks.
- **Logging Adjustments**: As algorithms adjust parameters or explore new solutions, they can store updated configurations, ensuring that each adjustment is logged.

These interactions ensure that the optimization process is well-documented and that historical data can be leveraged effectively, enhancing the decision-making process of optimization algorithms.


Design Considerations
---------------------

The design of the `datamanager` module is driven by the need to balance flexibility, performance, and scalability in handling optimization task data. This section outlines the key design considerations that guided the development of the module, ensuring that it meets the diverse needs of users working with complex optimization problems.

Modularity and Extensibility
****************************
The `datamanager` is designed with a modular architecture, where each major function—such as data storage, similarity search, and checkpointing—is encapsulated in a dedicated component. This modular design offers several advantages:

- **Separation of Concerns**: Each component handles a specific aspect of data management, allowing users to focus on individual functionalities without being overwhelmed by the entire system.
- **Ease of Maintenance**: With separate modules for each function, updates and bug fixes can be applied to specific areas without affecting the rest of the system, making maintenance simpler.
- **Extensibility**: New features or modifications can be added without disrupting the existing functionality. For example, adding support for a different database backend or a new similarity search technique can be achieved by extending the relevant module without needing to overhaul the entire system.

Performance Optimization
************************
Given the data-intensive nature of optimization tasks, performance was a critical consideration in the design of the `datamanager`. Several strategies were employed to ensure that the module can handle large datasets and complex queries efficiently:

- **Indexed Storage**: The use of indexes on key fields in the SQLite database—such as `table_name` and `problem_name`—ensures that searches and data retrievals are fast, even when the database grows in size.
- **Batch Data Insertion**: When storing large amounts of data, the `datamanager` supports batch insertion, reducing the overhead associated with frequent database writes. This approach minimizes transaction times and optimizes the overall data storage process.
- **Efficient Similarity Computation**: By using Locality-Sensitive Hashing (LSH) and MinHash, the module avoids the computational cost of pairwise comparisons in high-dimensional space, making similarity-based searches scalable even for large datasets.

These performance optimizations ensure that the `datamanager` remains responsive and efficient, providing a smooth experience for users dealing with large-scale optimization tasks.

Scalability and Data Volume Management
**************************************
As the `datamanager` is intended for use with potentially large datasets generated by optimization tasks, scalability was a key focus during its design:

- **Scalable Database Design**: The choice of SQLite as the initial database backend provides a lightweight and self-contained solution that is sufficient for many use cases. However, the design is abstract enough to allow for future migration to more powerful database systems like PostgreSQL if the need for greater scalability arises.
- **Incremental Data Storage**: The ability to store data incrementally during optimization tasks ensures that the database grows in a controlled manner, preventing sudden spikes in storage requirements. This is especially important for long-running tasks that generate large volumes of data over time.
- **Support for Data Archiving**: To prevent the database from becoming too large and unwieldy, the `datamanager` supports archiving of old or completed task data. This ensures that the active dataset remains manageable, while older records can be preserved externally if needed for historical analysis.

These design choices make the `datamanager` suitable for both small-scale experimentation and larger, more data-intensive optimization environments, adapting to the needs of different users.

Flexibility in Data Formats and Querying
****************************************
Flexibility is a hallmark of the `datamanager`'s design, ensuring that it can adapt to various data types and query requirements across different optimization problems:

- **Support for Multiple Data Formats**: The module supports the insertion and retrieval of data in various formats, including dictionaries, lists, pandas DataFrames, and NumPy arrays. This flexibility allows users to work with their preferred data structures without needing to conform to a rigid format.
- **Customizable Search and Query Mechanisms**: Users have the freedom to perform custom SQL queries directly on the database, allowing for advanced data manipulation and retrieval. This is especially useful when standard search methods do not meet specific analysis requirements.
- **Adaptable Metadata Design**: The `_metadata` table is designed to be extensible, allowing users to add new fields that are relevant to their particular optimization tasks. This ensures that the metadata stored for each task can evolve alongside the changing needs of the optimization process.

These aspects of flexibility make the `datamanager` an adaptable tool, suitable for a wide range of use cases and optimization frameworks.

Reliability and Data Integrity
******************************
Ensuring data integrity and reliability is critical when managing optimization task data. The `datamanager` includes several mechanisms to safeguard against data loss and ensure consistency:

- **Atomic Transactions**: The use of atomic transactions in database operations ensures that data is written consistently, reducing the risk of data corruption during inserts, updates, or deletions.
- **Checkpointing for Data Safety**: The checkpointing feature serves as a safeguard, allowing users to restore a task to a previous state if issues occur, ensuring that progress is not permanently lost.
- **Data Validation**: Basic validation checks are performed before data is inserted into the database, ensuring that required fields are present and data types are correct. This prevents invalid data from entering the system and causing errors during later stages of optimization.

With these mechanisms in place, the `datamanager` is designed to maintain the reliability and integrity of the data it manages, ensuring that users can trust it as a robust data management solution for their optimization tasks.

The design considerations of the `datamanager` reflect a balance between flexibility, performance, and reliability, making it a well-rounded choice for managing the complex and evolving data needs of optimization tasks. Its modular structure, scalability, and focus on data integrity ensure that it can adapt to different challenges and provide consistent value in optimization scenarios.

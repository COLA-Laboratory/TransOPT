import React from "react";
import {
  Table,
} from "reactstrap";

function TaskTable({tasks}) {
    // console.log(tasks);
    return (
        <Table lg={12} md={12} sm={12} striped>
            <thead>
                <tr className="fs-sm">
                <th className="hidden-sm-down">#</th>
                <th className="hidden-sm-down">Name</th>
                <th className="hidden-sm-down">Num_vars</th>
                <th className="hidden-sm-down">Num_objs</th>
                <th className="hidden-sm-down">Fidelity</th>
                <th className="hidden-sm-down">workloads</th>
                <th className="hidden-sm-down">budget_type</th>
                <th className="hidden-sm-down">budget</th>
                </tr>
            </thead>
            <tbody>
                {tasks.map((task, index) => (
                    <tr key={index}>
                        <td>{index+1}</td>
                        <td>{task.name}</td>
                        <td>{task.num_vars}</td>
                        <td>{task.num_objs}</td>
                        <td>{task.fidelity}</td>
                        <td>{task.workloads}</td>
                        <td>{task.budget_type}</td>
                        <td>{task.budget}</td>
                    </tr>
                ))}
            </tbody>
        </Table>
    );
}

export default TaskTable;
import React from "react";
import {
  Row,
  Col,
} from "reactstrap";

import s from "./problem.module.scss"

import Widget from "../../components/Widget/Widget";

import SelectTask from "./component/SelectTask";
import TaskTable from "./component/TaskTable";


class Problem extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      TasksData: [],
      tasks: [],
    };
  }

  updateTable = (newTasks) => {
    this.setState({ tasks: newTasks });
  }

  render() {
    if (this.state.TasksData.length === 0) {
      const messageToSend = {
        action: 'ask for basic information',
      }
      fetch('http://localhost:5000/api/configuration/basic_information', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(messageToSend),
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        } 
        return response.json();
      })
      .then(data => {
        console.log('Message from back-end:', data);
        this.setState({ TasksData: data.TasksData });
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });

      fetch('http://localhost:5000/api/RunPage/get_info', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(messageToSend),
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        } 
        return response.json();
      })
      .then(data => {
        console.log('Configuration infomation from back-end:', data);
        this.setState({ tasks: data.tasks });
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });

      return (
        <div className={s.root}>
          <h1 className="page-title">
            <span className="fw-semi-bold">Specify Problem</span>
          </h1>
        </div>
      )
    } else {
      return (
        <div className={s.root}>
          <h1 className="page-title">
            <span className="fw-semi-bold">Specify Problem</span>
          </h1>
            <Row>
              <Col lg={12} sm={12}>
                <Widget>
                  <SelectTask data={this.state.TasksData} updateTable={this.updateTable}/>
                </Widget>
              </Col>
              <Col lg={12} sm={12}>
                <Widget
                  title={
                    <h5>
                      <span className="fw-semi-bold">Problem Information</span>
                    </h5>
                  }
                  collapse
                >
                  <TaskTable tasks={this.state.tasks} />
                </Widget>
              </Col>
            </Row>
        </div>
      );
    }
  }
}

export default Problem;
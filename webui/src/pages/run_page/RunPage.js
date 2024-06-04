import React from "react";

import { Row, Col } from "reactstrap";

import s from "./RunPage.module.scss";
import Widget from "../../components/Widget/Widget";

import Run from "./component/Run"
import RunProgress from "./component/RunProgress"
import TaskTable from "./component/TaskTable";
import OptTable from "./component/OptTable";
import DataTable from "./component/DataTable";


class RunPage extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      get_info: false,
      tasks: [],
      optimizer: {},
      datasets: {},
    };
  }

  render() { 
    // If first time rendering, then render the default task
    // If not, then render the task that was clicked
    if (this.state.get_info === false) {
      // TODO: ask for task list from back-end
      const messageToSend = {
        action: 'ask for information',
      }
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
        this.setState({ get_info: true,  
                        tasks: data.tasks,
                        optimizer: data.optimizer,
                        datasets: data.datasets});
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });

      
      // Set the default task as the first task in the list
      return (
        <div className={s.root}>
          <h1 className="page-title">
            <span className="fw-semi-bold">Run</span>
          </h1>
        </div>
      )
    } else {

      return (
        <div className={s.root}>
          <h1 className="page-title">
          <span className="fw-semi-bold">Run Experiments</span>
          </h1>
          <div>
            <Row>
              <Col lg={12} xs={12}>
                <Widget
                  title={
                    <h5>
                      <span className="fw-semi-bold">Experiment Set-up</span>
                    </h5>
                  }
                  collapse
                >
                  <h4>
                    Problems
                  </h4>
                  <TaskTable tasks={this.state.tasks} />
                  <h4>
                    Algorithm
                  </h4>
                  <OptTable optimizer={this.state.optimizer} />
                  <h4>
                    Data
                  </h4>
                  <DataTable datasets={this.state.datasets} optimizer={this.state.optimizer}/>
                  <Run />
                  <RunProgress />
                </Widget>
              </Col>
            </Row>
          </div>
        </div>
      );
    }
  }

}

export default RunPage;

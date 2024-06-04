import React from "react";

import {
  Row,
  Col,
} from "reactstrap";

import Widget from "../../components/Widget/Widget";

import s from "./Comparison.module.scss";

import Box from "./charts/Box";
import Trajectory from "./charts/Trajectory";
import SelectTask from "./component/SelectTask"
import { Skeleton } from "antd";

class Comparison extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isFirst: true,
      selections: {},
      BoxData: {},
      TrajectoryData: [],
    };
  }

  handleClick = (values) => {
    console.log("Tasks:", values.Tasks)
    const messageToSend = values.Tasks.map(task => ({
      TaskName: task.TaskName || '',
      NumObjs: task.NumObjs || '',
      NumVars: task.NumVars || '',
      Fidelity: task.Fidelity || '',
      Workload: task.Workload || '',
      Seed: task.Seed || '',
      Refiner: task.Refiner || '',
      Sampler: task.Sampler || '',
      Pretrain: task.Pretrain || '',
      Model: task.Model || '',
      ACF: task.ACF || '',
      Normalizer: task.Normalizer || ''
    }));
    fetch('http://localhost:5000/api/comparison/choose_task', {
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
      // console.log('Message from back-end:', data);
      this.setState({ BoxData: data.BoxData, TrajectoryData: data.TrajectoryData });
    })
    .catch((error) => {
      console.error('Error sending message:', error);
    });
  }

  render() {
    if (this.state.isFirst) {
      const messageToSend = {
        message: 'ask for selections',
      }
      fetch('http://localhost:5000/api/comparison/selections', {
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
        this.setState({ selections: data , isFirst: false});
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });

      return (
        <div className={s.root}>
          <h1 className="page-title">
            Report - <span className="fw-semi-bold">Comparison</span>
          </h1>
        </div>
      )
    } else {
      return (
        <div className={s.root}>
          <h1 className="page-title">
            Report - <span className="fw-semi-bold">Comparison</span>
          </h1>
          <div>
            <Row>
              <Col lg={12} xs={12}>
                <Widget
                  title={
                    <h5>
                      <span className="fw-semi-bold">Choose Tasks</span>
                    </h5>
                  }
                  collapse
                >
                  <SelectTask selections={this.state.selections} handleClick={this.handleClick}/>
                </Widget>
              </Col>
              <Col lg={12} xs={12}>
                <Row>
                  <Col lg={6} xs={12}>
                    <Widget
                      title={
                        <h5>
                          <span className="fw-semi-bold">Trajectory</span>
                        </h5>
                      }
                      collapse
                    >
                      <Trajectory TrajectoryData={this.state.TrajectoryData}/>
                    </Widget>
                  </Col>
                  <Col lg={6} xs={12}>
                    <Widget
                      title={
                        <h5>
                          <span className="fw-semi-bold">Box</span>
                        </h5>
                      }
                      collapse
                    > 
                      <Box BoxData={this.state.BoxData}/>
                    </Widget>
                  </Col>
                </Row>
              </Col>
            </Row>
          </div>
        </div>
      );
    }
  }
}

export default Comparison;

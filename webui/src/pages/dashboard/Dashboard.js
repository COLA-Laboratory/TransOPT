import React from "react";

import { Row, Col, Button } from "reactstrap";

import s from "./dashboard.module.scss";
import Widget from "../../components/Widget/Widget";

import Trajectory from "./charts/Trajectory";
import Radar from "./charts/Radar";
import Scatter from "./charts/Scatter";
import Bar from "./charts/Bar";
import Importance from "./charts/Importance";


class Dashboard extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      selectedTaskIndex: -1,
      tasksInfo: [],
      ScatterData: [],
      TrajectoryData: []
    };
  }

  // Select the corresponding task to display
  handleTaskClick = (index) => {
    console.log(index)
    this.setState({ selectedTaskIndex: index });
    const messageToSend = {
      taskname:this.state.tasksInfo[this.state.selectedTaskIndex].problem_name,
    }
    fetch('http://localhost:5000/api/Dashboard/charts', {
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
      this.setState({
        // BarData: data.BarData,
        // RadarData: data.RadarData,
        ScatterData: data.ScatterData,
        TrajectoryData: data.TrajectoryData
      })
    })
    .catch((error) => {
      console.error('Error sending message:', error);
    });
  }

  componentDidMount() {
    // 开始定时调用 fetchData 函数
    this.intervalId = setInterval(this.fetchData, 2000);
  }

  componentWillUnmount() {
    // 清除定时器，以防止内存泄漏
    clearInterval(this.intervalId);
  }

  fetchData = async () => {
    try {
      const messageToSend = {
        taskname:this.state.tasksInfo[this.state.selectedTaskIndex].problem_name,
      }
      const response = await fetch('http://localhost:5000/api/Dashboard/trajectory', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(messageToSend)
      });
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json();
      console.log('Data from server:', data);
      // 在这里处理从服务器获取的数据
      this.setState({
        // BarData: data.BarData,
        // RadarData: data.RadarData,
        // ScatterData: data.ScatterData,
        TrajectoryData: data.TrajectoryData
      })
      // console.log('State:', this.state.BarData)
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  render() { 
    // If first time rendering, then render the default task
    // If not, then render the task that was clicked
    if (this.state.selectedTaskIndex === -1) {
      // TODO: ask for task list from back-end
      const messageToSend = {
        action: 'ask for tasks information',
      }
      fetch('http://localhost:5000/api/Dashboard/tasks', {
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
        this.setState({ selectedTaskIndex: 0,  tasksInfo: data });
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });

      
      // Set the default task as the first task in the list
      return (
        <div className={s.root}>
          <h1 className="page-title">
            Dashboard - <span className="fw-semi-bold">Tasks</span>
          </h1>
        </div>
      )
    } else {

      return (
        <div className={s.root}>
          <h1 className="page-title">
           <span className="fw-semi-bold">Dashboard</span>
          </h1>
          <div>
            <Row>
              <Col lg={3} xs={4}>
                <Row>
                  <Col lg={12} xs={12}>
                    <Widget
                      title={
                        <h5>
                          Choose <span className="fw-semi-bold">Dataset</span>
                        </h5>
                      }
                      collapse
                    >
                    <div style={{ overflowY: 'auto', maxHeight: '300px' }}>
                    {this.state.tasksInfo.map((task, index) => (
                      <Button
                        key={index}
                        className={s.btn}
                        onClick={() => this.handleTaskClick(index)}
                        >
                          {task.problem_name}
                        </Button>
                    ))}
                    </div>
                    </Widget>
                  </Col>
                  <Col lg={12} xs={12}>
                    <Widget
                      title={
                        <h5>
                          <span className="fw-semi-bold">Information</span>
                        </h5>
                      }
                      collapse
                    >
                      <div style={{ overflowY: 'auto', maxHeight: '650px' }}>
                      <h4><strong>Task</strong></h4>
                      <ul>
                        <li><h5><span className="fw-semi-bold">Problem Name</span>: {this.state.tasksInfo[this.state.selectedTaskIndex].problem_name}</h5></li>
                        <li><h5><span className="fw-semi-bold">Variable num</span>: {this.state.tasksInfo[this.state.selectedTaskIndex].dim}</h5></li>
                        <li><h5><span className="fw-semi-bold">Objective num</span>: {this.state.tasksInfo[this.state.selectedTaskIndex].obj}</h5></li>
                        <li><h5><span className="fw-semi-bold">Fidelity</span>: {this.state.tasksInfo[this.state.selectedTaskIndex].fidelity}</h5></li>
                        <li><h5><span className="fw-semi-bold">Workloads</span>: {this.state.tasksInfo[this.state.selectedTaskIndex].workloads}</h5></li>
                        <li><h5><span className="fw-semi-bold">Budget type</span>: {this.state.tasksInfo[this.state.selectedTaskIndex].budget_type}</h5></li>
                        <li><h5><span className="fw-semi-bold">Budget</span>: {this.state.tasksInfo[this.state.selectedTaskIndex].budget}</h5></li>
                        <li><h5><span className="fw-semi-bold">Seeds</span>: {this.state.tasksInfo[this.state.selectedTaskIndex].seeds}</h5></li>
                      </ul>
                      <h4 className="mt-5"><strong>Algorithm</strong></h4>
                      <ul>
                        <li><h5><span className="fw-semi-bold">Narrow Search Space</span>: {this.state.tasksInfo[this.state.selectedTaskIndex].SpaceRefiner}</h5></li>
                        <li><h5><span className="fw-semi-bold">Initialization</span>: {this.state.tasksInfo[this.state.selectedTaskIndex].Sampler}</h5></li>
                        <li><h5><span className="fw-semi-bold">Pre-train</span>: {this.state.tasksInfo[this.state.selectedTaskIndex].Pretrain}</h5></li>
                        <li><h5><span className="fw-semi-bold">Surrogate Model</span>: {this.state.tasksInfo[this.state.selectedTaskIndex].Model}</h5></li>
                        <li><h5><span className="fw-semi-bold">Acquisition Function</span>: {this.state.tasksInfo[this.state.selectedTaskIndex].ACF}</h5></li>
                        <li><h5><span className="fw-semi-bold">DatasetSelector</span>: {this.state.tasksInfo[this.state.selectedTaskIndex].DatasetSelector}</h5></li>
                        <li><h5><span className="fw-semi-bold">Normalizer</span>: {this.state.tasksInfo[this.state.selectedTaskIndex].Normalizer}</h5></li>
                      </ul>
                      <h4 className="mt-5"><strong>Data List</strong></h4>
                      <ul>
                        <li><h5><span className="fw-semi-bold">Narrow Search Space</span>:</h5></li>
                        <div>
                          <ul>
                            {this.state.tasksInfo[this.state.selectedTaskIndex].metadata.SpaceRefiner.map((dataset, index) => (
                              <li key={index}><h5>{dataset}</h5></li>
                            ))}
                          </ul>
                        </div>
                        <li><h5><span className="fw-semi-bold">Initialization</span>:</h5></li>
                        <div>
                          <ul>
                            {this.state.tasksInfo[this.state.selectedTaskIndex].metadata.Sampler.map((dataset, index) => (
                              <li key={index}><h5>{dataset}</h5></li>
                            ))}
                          </ul>
                        </div>
                        <li><h5><span className="fw-semi-bold">Pre-train</span>:</h5></li>
                        <div>
                          <ul>
                            {this.state.tasksInfo[this.state.selectedTaskIndex].metadata.Pretrain.map((dataset, index) => (
                              <li key={index}><h5>{dataset}</h5></li>
                            ))}
                          </ul>
                        </div>
                        <li><h5><span className="fw-semi-bold">Surrogate Model</span>:</h5></li>
                        <div>
                          <ul>
                            {this.state.tasksInfo[this.state.selectedTaskIndex].metadata.Model.map((dataset, index) => (
                              <li key={index}><h5>{dataset}</h5></li>
                            ))}
                          </ul>
                        </div>
                        <li><h5><span className="fw-semi-bold">Acquisition Function</span>:</h5></li>
                        <div>
                          <ul>
                            {this.state.tasksInfo[this.state.selectedTaskIndex].metadata.ACF.map((dataset, index) => (
                              <li key={index}><h5>{dataset}</h5></li>
                            ))}
                          </ul>
                        </div>
                        <li><h5><span className="fw-semi-bold">Normalizer</span>:</h5></li>
                        <div>
                          <ul>
                            {this.state.tasksInfo[this.state.selectedTaskIndex].metadata.Normalizer.map((dataset, index) => (
                              <li key={index}><h5>{dataset}</h5></li>
                            ))}
                          </ul>
                        </div>
                      </ul>
                      </div>
                    </Widget>
                  </Col>
                </Row>
              </Col>
              <Col lg={9} xs={8}>
                <Row>
                  <Col lg={12} xs={12}>
                    <Widget
                      title={
                        <h5>
                          <span className="fw-semi-bold">Convergence Curve</span>
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
                            <span className="fw-semi-bold">Footprint</span>
                        </h5>
                      }
                      collapse
                    >
                      <Scatter ScatterData={this.state.ScatterData} />
                    </Widget>
                  </Col>
                  <Col lg={6} xs={12}>
                    <Widget
                      title={
                        <h5>
                          <span className="fw-semi-bold">Variables Network</span>
                        </h5>
                      }
                      collapse
                    >
                      <Importance />
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

export default Dashboard;

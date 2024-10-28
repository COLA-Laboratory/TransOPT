import React from "react";

import {
  Row,
  Col,
} from "reactstrap";

import TitleCard from "../../components/Cards/TitleCard"
import LineChart from './components/LineChart'


import Box from "./charts/Box";
import Trajectory from "./charts/Trajectory";
import SelectTask from "./components/SelectTask.js"
import { Skeleton } from "antd";

class Analytics extends React.Component {
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
    fetch('http://localhost:5001/api/comparison/choose_task', {
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
      fetch('http://localhost:5001/api/comparison/selections', {
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

    } else {
      return (
        <div>
            <div>
                <TitleCard
                  title={
                    <h5>
                      <span className="fw-semi-bold">Filter</span>
                    </h5>
                  }
                  collapse
                >
                  <SelectTask selections={this.state.selections} handleClick={this.handleClick}/>
                </TitleCard>

            <div className="grid mt-4 grid-cols-1 lg:grid-cols-[50%_50%] gap-6">

                <LineChart TrajectoryData={this.state.TrajectoryData} />


                <TitleCard
                    title={
                    <h5>
                        <span className="fw-semi-bold">Box</span>
                    </h5>
                    }
                    collapse
                > 
                    <Box BoxData={this.state.BoxData}/>
                </TitleCard>
          </div>
          </div>
        </div>
      );
    }
  }
}

export default Analytics;

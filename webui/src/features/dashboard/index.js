import React from "react";

import { Row, Col, Button } from "reactstrap";
import {Input, Modal } from "antd";  
import TitleCard from "../../components/Cards/TitleCard"


import LineChart from './components/LineChart'
import BarChart from './components/BarChart'
// import Footprint from "./components/Footprint";
import Footprint from "./components/ScatterChart";



class Dashboard extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      selectedTaskIndex: -1,
      tasksInfo: [],
      ScatterData: [],
      TrajectoryData: [],
      isModalVisible: false,  // 控制Modal显示
      errorMessage: ""  // 存储输入的错误信息
    };
  }

  // Select the corresponding task to display
  handleTaskClick = (index) => {
    console.log(index)
    this.setState({ selectedTaskIndex: index });
    const messageToSend = {
      taskname:this.state.tasksInfo[this.state.selectedTaskIndex].problem_name,
    }
    fetch('http://localhost:5001/api/Dashboard/charts', {
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

  // 处理按钮点击事件
  showModal = () => {
    this.setState({ isModalVisible: true });
  };

  handleOk = () => {
    console.log(this.state.errorMessage);
    this.setState({ isModalVisible: false });

    const messageToSend = {
    errorMessage: this.state.errorMessage
  };

  fetch("http://localhost:5001/api/Dashboard/errorsubmit", {  // 根据实际的API端点进行调整
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(messageToSend),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return response.json();
    })
    .then((data) => {
      console.log("Message sent successfully:", data);
      this.setState({ isModalVisible: false, errorMessage: "" });
    })
    .catch((error) => {
      console.error("Error sending message:", error);
    });
};


  


  handleCancel = () => {
    this.setState({ isModalVisible: false });
  };

  handleInputChange = (e) => {
    this.setState({ errorMessage: e.target.value });
  };

  componentDidMount() {
    // 开始定时调用 fetchData 函数
    this.intervalId = setInterval(this.fetchData, 1000000);
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
      const response = await fetch('http://localhost:5001/api/Dashboard/trajectory', {
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
        ScatterData: data.ScatterData,
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
      fetch('http://localhost:5001/api/Dashboard/tasks', {
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
        <div>
          <h1 className="page-title">
            Dashboard - <span className="fw-semi-bold">Tasks</span>
          </h1>
        </div>
      )
    } 
    else {

        return (
            <>
            <TitleCard
                title={
                    <h5>
                    <span className="fw-semi-bold">Choose Dataset</span>
                    </h5>
                    }
                    collapse
            >

                <div className="grid mt-4 grid-cols-1 lg:grid-cols-[20%_80%] gap-6">

                <div style={{ overflowY: 'auto', maxHeight: "400px" }}>
                    {this.state.tasksInfo.map((task, index) => (
                        <Button
                        key={index}
                        onClick={() => this.handleTaskClick(index)}
                        style={{ backgroundColor: 'rgba(53, 162, 235, 0.5)', color: '#000000' }} // 自定义背景色和文字颜色

                        >
                        {task.problem_name}
                        
                        </Button>
                    ))}
                </div>
                <div style={{ overflowY: 'auto', maxHeight: '400px', padding: '10px', border: '1px solid #ddd', borderRadius: '8px', backgroundColor: '#f9f9f9' }}>
                
                <section style={{ marginBottom: '20px', borderBottom: '1px solid #e0e0e0', paddingBottom: '10px' }}>
                    <h4 style={{ color: '#333', marginBottom: '10px', fontSize: '1.2em', fontWeight: 'bold' }}>
                        Problem Information
                    </h4>
                    <ul style={{ listStyle: 'none', padding: 0, lineHeight: '1.6' }}>
                        <li style={{ marginBottom: '8px', fontSize: '0.95em' }}>
                        <strong>Problem Name:</strong> {this.state.tasksInfo[this.state.selectedTaskIndex].problem_name}
                        </li>
                        <li style={{ marginBottom: '8px', fontSize: '0.95em' }}>
                        <strong>Variable num:</strong> {this.state.tasksInfo[this.state.selectedTaskIndex].dim},&nbsp;
                        <strong>Objective num:</strong> {this.state.tasksInfo[this.state.selectedTaskIndex].obj},&nbsp;
                        <strong>Seeds:</strong> {this.state.tasksInfo[this.state.selectedTaskIndex].seeds},&nbsp;
                        <strong>Budget type:</strong> {this.state.tasksInfo[this.state.selectedTaskIndex].budget_type},&nbsp;
                        <strong>Budget:</strong> {this.state.tasksInfo[this.state.selectedTaskIndex].budget},&nbsp;
                        <strong>Workloads:</strong> {this.state.tasksInfo[this.state.selectedTaskIndex].workloads},&nbsp;
                        <strong>Fidelity:</strong> {this.state.tasksInfo[this.state.selectedTaskIndex].fidelity}
                        </li>
                    </ul>
                </section>


                <section style={{ marginBottom: '20px', borderBottom: '1px solid #e0e0e0', paddingBottom: '10px' }}>
                    <h4 style={{ color: '#333', marginBottom: '10px', fontSize: '1.2em', fontWeight: 'bold' }}>
                        Algorithm Objects
                    </h4>
                    <ul style={{ listStyle: 'none', padding: 0, lineHeight: '1.6' }}>
                        <li style={{ marginBottom: '8px', fontSize: '0.95em' }}>
                        <strong>Narrow Search Space:</strong> {this.state.tasksInfo[this.state.selectedTaskIndex].SpaceRefiner},&nbsp;
                        <strong>Initialization:</strong> {this.state.tasksInfo[this.state.selectedTaskIndex].Sampler},&nbsp;
                        <strong>Pre-train:</strong> {this.state.tasksInfo[this.state.selectedTaskIndex].Pretrain},&nbsp;
                        <strong>Surrogate Model:</strong> {this.state.tasksInfo[this.state.selectedTaskIndex].Model},&nbsp;
                        <strong>Acquisition Function:</strong> {this.state.tasksInfo[this.state.selectedTaskIndex].ACF},&nbsp;
                        <strong>Normalizer:</strong> {this.state.tasksInfo[this.state.selectedTaskIndex].Normalizer}
                        </li>
                        <li style={{ marginBottom: '8px', fontSize: '0.95em' }}>
                        <strong>DatasetSelector:</strong> {this.state.tasksInfo[this.state.selectedTaskIndex].DatasetSelector}
                        </li>
                    </ul>
                </section>

                <section style={{ marginBottom: '20px', paddingBottom: '10px' }}>
                    <h4 style={{ color: '#333', marginBottom: '10px', fontSize: '1.2em', fontWeight: 'bold' }}>
                        Auxilliary Data List
                    </h4>

                    <div style={{ marginBottom: '10px', fontSize: '0.95em' }}>
                        <strong>Narrow Search Space:</strong>
                        <ul style={{ listStyle: 'square', paddingLeft: '20px', lineHeight: '1.6' }}>
                        {this.state.tasksInfo[this.state.selectedTaskIndex].metadata.SpaceRefiner.map((dataset, index) => (
                            <li key={index}>{dataset}</li>
                        ))}
                        </ul>
                    </div>

                    <div style={{ marginBottom: '10px', fontSize: '0.95em' }}>
                        <strong>Initialization:</strong>
                        <ul style={{ listStyle: 'square', paddingLeft: '20px', lineHeight: '1.6' }}>
                        {this.state.tasksInfo[this.state.selectedTaskIndex].metadata.Sampler.map((dataset, index) => (
                            <li key={index}>{dataset}</li>
                        ))}
                        </ul>
                    </div>

                    <div style={{ marginBottom: '10px', fontSize: '0.95em' }}>
                        <strong>Pre-train:</strong>
                        <ul style={{ listStyle: 'square', paddingLeft: '20px', lineHeight: '1.6' }}>
                        {this.state.tasksInfo[this.state.selectedTaskIndex].metadata.Pretrain.map((dataset, index) => (
                            <li key={index}>{dataset}</li>
                        ))}
                        </ul>
                    </div>

                    <div style={{ marginBottom: '10px', fontSize: '0.95em' }}>
                        <strong>Surrogate Model:</strong>
                        <ul style={{ listStyle: 'square', paddingLeft: '20px', lineHeight: '1.6' }}>
                        {this.state.tasksInfo[this.state.selectedTaskIndex].metadata.Model.map((dataset, index) => (
                            <li key={index}>{dataset}</li>
                        ))}
                        </ul>
                    </div>

                    <div style={{ marginBottom: '10px', fontSize: '0.95em' }}>
                        <strong>Acquisition Function:</strong>
                        <ul style={{ listStyle: 'square', paddingLeft: '20px', lineHeight: '1.6' }}>
                        {this.state.tasksInfo[this.state.selectedTaskIndex].metadata.ACF.map((dataset, index) => (
                            <li key={index}>{dataset}</li>
                        ))}
                        </ul>
                    </div>

                    <div style={{ marginBottom: '10px', fontSize: '0.95em' }}>
                        <strong>Normalizer:</strong>
                        <ul style={{ listStyle: 'square', paddingLeft: '20px', lineHeight: '1.6' }}>
                        {this.state.tasksInfo[this.state.selectedTaskIndex].metadata.Normalizer.map((dataset, index) => (
                            <li key={index}>{dataset}</li>
                        ))}
                        </ul>
                    </div>
                </section>

                    </div>
            </div>
            </TitleCard>


              <div className="grid lg:grid-cols-3 mt-4 grid-cols-1 gap-6">
                <LineChart  TrajectoryData={this.state.TrajectoryData}/>
                <BarChart ImportanceData={this.state.Importance}/>
                <Footprint ScatterData={this.state.ScatterData}/>
              </div>
          

            </>
          );          
    }
  }
}

export default Dashboard;

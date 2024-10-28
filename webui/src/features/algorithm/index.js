import React from "react";

import TitleCard from "../../components/Cards/TitleCard"

import SelectPlugins from "./components/SelectPlugin";
import OptTable from "./components/OptTable";


class Algorithm extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      TasksData: [],
      SpaceRefiner: [],
      Sampler: [],
      Pretrain: [],
      Model: [],
      ACF: [],
      DataSelector: [],
      Normalizer: [],
      optimizer: {},
    };
  }

  updateTable = (newOptimizer) => {
    this.setState({ optimizer: newOptimizer });
  }

  render() {
    if (this.state.TasksData.length === 0) {
      const messageToSend = {
        action: 'ask for basic information',
      }
      fetch('http://localhost:5001/api/configuration/basic_information', {
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
        this.setState({ TasksData: data.TasksData,
                        SpaceRefiner: data.SpaceRefiner,
                        Sampler: data.Sampler,
                        Pretrain: data.Pretrain,
                        Model: data.Model,
                        ACF: data.ACF,
                        DataSelector: data.DataSelector,
                        Normalizer: data.Normalizer,
                      });
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });

      fetch('http://localhost:5001/api/RunPage/get_info', {
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
        this.setState({ optimizer: data.optimizer });
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });
    } else {
      return (
        <div>
          <div className="grid mt-4 grid-cols-1 lg:grid-cols-[50%_50%] gap-6">
                <TitleCard>
                  <SelectPlugins SpaceRefiner={this.state.SpaceRefiner}
                                    Sampler={this.state.Sampler}
                                    Pretrain={this.state.Pretrain}
                                    Model={this.state.Model}
                                    ACF={this.state.ACF}
                                    DataSelector={this.state.DataSelector}
                                    Normalizer={this.state.Normalizer}
                                    updateTable={this.updateTable}
                  />
                </TitleCard>

                <TitleCard
                  title={
                    <h5>
                      <span className="fw-semi-bold">Composition</span>
                    </h5>
                  }
                  collapse
                >
                  <OptTable optimizer={this.state.optimizer} />
                </TitleCard>
            </div>
        </div>
      );
    }
  }
}

export default Algorithm;
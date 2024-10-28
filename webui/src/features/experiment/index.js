import React from "react";

import TitleCard from "../../components/Cards/TitleCard"


import SelectTask from "./components/SelectTask";
import SelectAlgorithm from "./components/SelectAlgorithm";
import SearchData from "./components/SearchData";
import SelectData from "./components/SelectData";

class Experiment extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      TasksData: [],
      tasks: [],
      SpaceRefiner: [],
      Sampler: [],
      Pretrain: [],
      Model: [],
      ACF: [],
      DataSelector: [],
      Normalizer: [],
      optimizer: {},

      get_info: false,
      DatasetData: {"isExact": false, "datasets": []},
      SpaceRefinerDataSelector: "",
      SpaceRefinerDataSelectorParameters: "",
      SamplerDataSelector: "",
      SamplerDataSelectorParameters: "",
      PretrainDataSelector: "",
      PretrainDataSelectorParameters: "",
      ModelDataSelector: "",
      ModelDataSelectorParameters: "",
      ACFDataSelector: "",
      ACFDataSelectorParameters: "",
      NormalizerDataSelector: "",
      NormalizerDataSelectorParameters: "",
      DatasetSelector: [],
    };
  }

  updateTaskTable = (newTasks) => {
    this.setState({ tasks: newTasks });
  }

  updateOptTable = (newOptimizer) => {
    this.setState({ optimizer: newOptimizer  });
  }


  updateDataTable = (newDatasets) => {
    console.log("newDatasets", newDatasets)
    const { object, DatasetSelector, parameter, datasets } = newDatasets;
    if (object === "Narrow Search Space") {
      this.setState({ SpaceRefiner: datasets, SpaceRefinerDataSelector: DatasetSelector, SpaceRefinerDataSelectorParameters: parameter})
    } else if (object === "Initialization") {
      this.setState({ Sampler: datasets, SamplerDataSelector: DatasetSelector, SamplerDataSelectorParameters: parameter})
    } else if (object === "Pre-train") {
      this.setState({ Pretrain: datasets, PretrainDataSelector: DatasetSelector, PretrainDataSelectorParameters: parameter})
    } else if (object === "Surrogate Model") {
      this.setState({ Model: datasets, ModelDataSelector: DatasetSelector, ModelDataSelectorParameters: parameter})
    } else if (object === "Acquisition Function") {
      this.setState({ ACF: datasets, ACFDataSelector: DatasetSelector, ACFDataSelectorParameters: parameter})
    } else if (object === "Normalizer") {
      this.setState({ Normalizer: datasets, NormalizerDataSelector: DatasetSelector, NormalizerDataSelectorParameters: parameter})
    }
  }

  set_dataset = (datasets) => {
    console.log(datasets)
    this.setState({ DatasetData: datasets })
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
                        tasks: data.tasks,
                        SpaceRefiner: data.SpaceRefiner,
                        Sampler: data.Sampler,
                        Pretrain: data.Pretrain,
                        Model: data.Model,
                        ACF: data.ACF,
                        DataSelector: data.DataSelector,
                        Normalizer: data.Normalizer,
                        get_info: true,  
                        SpaceRefinerDataSelector: data.SpaceRefinerDataSelector,
                        SpaceRefinerDataSelectorParameters: data.SpaceRefinerDataSelectorParameters,
                        SamplerDataSelector: data.SamplerDataSelector,
                        SamplerDataSelectorParameters: data.SamplerDataSelectorParameters,
                        PretrainDataSelector: data.PretrainDataSelector,
                        PretrainDataSelectorParameters: data.PretrainDataSelectorParameters,
                        ModelDataSelector: data.ModelDataSelector,
                        ModelDataSelectorParameters: data.ModelDataSelectorParameters,
                        ACFDataSelector: data.ACFDataSelector,
                        ACFDataSelectorParameters: data.ACFDataSelectorParameters,
                        NormalizerDataSelector: data.NormalizerDataSelector,
                        NormalizerDataSelectorParameters: data.NormalizerDataSelectorParameters,
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
        this.setState({ tasks: data.tasks, 
                        optimizer: data.optimizer});
                    })
                    .catch((error) => {
        console.error('Error sending message:', error);
      });


    } else {
      return (
        <div className="grid mt-4 w-[1200px] h-[800px] gap-6">

            <TitleCard >
                      <SelectTask data={this.state.TasksData} updateTable={this.updateTaskTable}/>
            </TitleCard>



            <TitleCard
              title={
                <h5>
                  <span className="fw-semi-bold">Build Algorithm</span>
                </h5>
              }
              collapse
            >
              <SelectAlgorithm SpaceRefiner={this.state.SpaceRefiner}
                                    Sampler={this.state.Sampler}
                                    Pretrain={this.state.Pretrain}
                                    Model={this.state.Model}
                                    ACF={this.state.ACF}
                                    DataSelector={this.state.DataSelector}
                                    Normalizer={this.state.Normalizer}
                                    updateTable={this.updateOptTable} />
            </TitleCard>        



            <TitleCard
              title={
                <h5>
                  <span className="fw-semi-bold">Customize Auxiliary Data</span>
                </h5>
              }
              collapse
            >
                  <SearchData set_dataset={this.set_dataset}/>
                  <p>
                    Choose the datasets you want to use in the experiment.
                  </p>
                  <SelectData DatasetData={this.state.DatasetData} updateTable={this.updateDataTable} DatasetSelector={this.state.DatasetSelector}/>
            </TitleCard>
            
        </div>
      );
    }
  }
}

export default Experiment;
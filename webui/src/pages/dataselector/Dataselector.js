import React from "react";
import {
  Row,
  Col,
} from "reactstrap";

import s from "./Dataselector.module.scss"

import Widget from "../../components/Widget/Widget";

import SelectData from "./component/SelectData";
import SearchData from "./component/SearchData"
import DataTable from "./component/DataTable";


class Dataselector extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      get_info: false,
      DatasetData: {"isExact": false, "datasets": []},
      SpaceRefiner: [],
      SpaceRefinerDataSelector: "",
      SpaceRefinerDataSelectorParameters: "",
      Sampler: [],
      SamplerDataSelector: "",
      SamplerDataSelectorParameters: "",
      Pretrain: [],
      PretrainDataSelector: "",
      PretrainDataSelectorParameters: "",
      Model: [],
      ModelDataSelector: "",
      ModelDataSelectorParameters: "",
      ACF: [],
      ACFDataSelector: "",
      ACFDataSelectorParameters: "",
      Normalizer: [],
      NormalizerDataSelector: "",
      NormalizerDataSelectorParameters: "",
      DatasetSelector: [],
    };
  }

  updateTable = (newDatasets) => {
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
                        SpaceRefiner: data.datasets.SpaceRefiner,
                        Sampler: data.datasets.Sampler,
                        Pretrain: data.datasets.Pretrain,
                        Model: data.datasets.Model,
                        ACF: data.datasets.ACF,
                        Normalizer: data.datasets.Normalizer,
                        SpaceRefinerDataSelector: data.optimizer.SpaceRefinerDataSelector,
                        SpaceRefinerDataSelectorParameters: data.optimizer.SpaceRefinerDataSelectorParameters,
                        SamplerDataSelector: data.optimizer.SamplerDataSelector,
                        SamplerDataSelectorParameters: data.optimizer.SamplerDataSelectorParameters,
                        PretrainDataSelector: data.optimizer.PretrainDataSelector,
                        PretrainDataSelectorParameters: data.optimizer.PretrainDataSelectorParameters,
                        ModelDataSelector: data.optimizer.ModelDataSelector,
                        ModelDataSelectorParameters: data.optimizer.ModelDataSelectorParameters,
                        ACFDataSelector: data.optimizer.ACFDataSelector,
                        ACFDataSelectorParameters: data.optimizer.ACFDataSelectorParameters,
                        NormalizerDataSelector: data.optimizer.NormalizerDataSelector,
                        NormalizerDataSelectorParameters: data.optimizer.NormalizerDataSelectorParameters,
                      });
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });

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
        this.setState({ DatasetSelector: data.DataSelector });
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });

      return (
        <div className={s.root}>
          <h1 className="page-title">
            <span className="fw-semi-bold">Select Data</span>
          </h1>
        </div>
      )
  } else {
      return (
        <div className={s.root}>
          <h1 className="page-title">
            <span className="fw-semi-bold">Select Data</span>
          </h1>
            <Row>
              <Col lg={12} sm={12}> 
                <Widget>
                  <SearchData set_dataset={this.set_dataset}/>
                  <p>
                    Choose the datasets you want to use in the experiment.
                  </p>
                  <SelectData DatasetData={this.state.DatasetData} updateTable={this.updateTable} DatasetSelector={this.state.DatasetSelector}/>
                </Widget>
              </Col>
              <Col lg={12} xs={12}>
                <Widget
                  title={
                    <h5>
                      <span className="fw-semi-bold">Selected Datasets</span>
                    </h5>
                  }
                  collapse
                >
                  <DataTable SpaceRefiner={this.state.SpaceRefiner} 
                              SpaceRefinerDataSelector={this.state.SpaceRefinerDataSelector}
                              SpaceRefinerDataSelectorParameters={this.state.SpaceRefinerDataSelectorParameters}
                              Sampler={this.state.Sampler} 
                              SamplerDataSelector={this.state.SamplerDataSelector}
                              SamplerDataSelectorParameters={this.state.SamplerDataSelectorParameters}
                              Pretrain={this.state.Pretrain} 
                              PretrainDataSelector={this.state.PretrainDataSelector}
                              PretrainDataSelectorParameters={this.state.PretrainDataSelectorParameters}
                              Model={this.state.Model} 
                              ModelDataSelector={this.state.ModelDataSelector}
                              ModelDataSelectorParameters={this.state.ModelDataSelectorParameters}
                              ACF={this.state.ACF} 
                              ACFDataSelector={this.state.ACFDataSelector}
                              ACFDataSelectorParameters={this.state.ACFDataSelectorParameters}
                              Normalizer={this.state.Normalizer}
                              NormalizerDataSelector={this.state.NormalizerDataSelector}
                              NormalizerDataSelectorParameters={this.state.NormalizerDataSelectorParameters}
                  />
                </Widget>
              </Col>
            </Row>
        </div>
      );
    }
  }
}

export default Dataselector;
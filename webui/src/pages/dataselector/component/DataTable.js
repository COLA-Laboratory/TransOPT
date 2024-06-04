import React from "react";
import {
  Table,
} from "reactstrap";

function DataTable({ SpaceRefiner, SpaceRefinerDataSelector, SpaceRefinerDataSelectorParameters,
    Sampler, SamplerDataSelector, SamplerDataSelectorParameters,
    Pretrain, PretrainDataSelector, PretrainDataSelectorParameters,
    Model, ModelDataSelector, ModelDataSelectorParameters,
    ACF, ACFDataSelector, ACFDataSelectorParameters,
    Normalizer, NormalizerDataSelector, NormalizerDataSelectorParameters,
}) {
    console.log("SpaceRefiner",SpaceRefiner);
    return (
        <Table lg={12} md={12} sm={12} striped>
            <thead>
                <tr className="fs-sm">
                <th className="hidden-sm-down">#</th>
                <th className="hidden-sm-down">DataSelector</th>
                <th className="hidden-sm-down">Parameters</th>
                <th className="hidden-sm-down">Datasets</th>
                </tr>
            </thead>
            <tbody>
                <tr key="SpaceRefiner">
                    <td>Narrow Search Space</td>
                    <td>{SpaceRefinerDataSelector}</td>
                    <td>{SpaceRefinerDataSelectorParameters}</td>
                    <td>{SpaceRefiner.join(', ')}</td>
                </tr>
                <tr key="Sampler">
                    <td>Initialization</td>
                    <td>{SamplerDataSelector}</td>
                    <td>{SamplerDataSelectorParameters}</td>
                    <td>{Sampler.join(', ')}</td>
                </tr>
                <tr key="Pretrain">
                    <td>Pre-train</td>
                    <td>{PretrainDataSelector}</td>
                    <td>{PretrainDataSelectorParameters}</td>
                    <td>{Pretrain.join(', ')}</td>
                </tr>
                <tr key="Model">
                    <td>Surrogate Model</td>
                    <td>{ModelDataSelector}</td>
                    <td>{ModelDataSelectorParameters}</td>
                    <td>{Model.join(', ')}</td>
                </tr>
                <tr key="ACF">
                    <td>Acquisition Function</td>
                    <td>{ACFDataSelector}</td>
                    <td>{ACFDataSelectorParameters}</td>
                    <td>{ACF.join(', ')}</td>
                </tr>
                <tr key="Normalizer">
                    <td>Normalizer</td>
                    <td>{NormalizerDataSelector}</td>
                    <td>{NormalizerDataSelectorParameters}</td>
                    <td>{Normalizer.join(', ')}</td>
                </tr>
            </tbody>
        </Table>
    );
}

export default DataTable;
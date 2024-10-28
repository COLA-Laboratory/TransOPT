import React from "react";
import {
  Table,
} from "reactstrap";

function DataTable({datasets, optimizer}) {
    // console.log("datasets",datasets);
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
                    <td>{optimizer.SpaceRefinerDataSelector}</td>
                    <td>{optimizer.SpaceRefinerDataSelectorParameters}</td>
                    <td>{datasets.SpaceRefiner.join(', ')}</td>
                </tr>
                <tr key="Sampler">
                    <td>Initialization</td>
                    <td>{optimizer.SamplerDataSelector}</td>
                    <td>{optimizer.SamplerDataSelectorParameters}</td>
                    <td>{datasets.Sampler.join(', ')}</td>
                </tr>
                <tr key="Pretrain">
                    <td>Pre-train</td>
                    <td>{optimizer.PretrainDataSelector}</td>
                    <td>{optimizer.PretrainDataSelectorParameters}</td>
                    <td>{datasets.Pretrain.join(', ')}</td>
                </tr>
                <tr key="Model">
                    <td>Surrogate Model</td>
                    <td>{optimizer.ModelDataSelector}</td>
                    <td>{optimizer.ModelDataSelectorParameters}</td>
                    <td>{datasets.Model.join(', ')}</td>
                </tr>
                <tr key="ACF">
                    <td>Acquisition Function</td>
                    <td>{optimizer.ACFDataSelector}</td>
                    <td>{optimizer.ACFDataSelectorParameters}</td>
                    <td>{datasets.ACF.join(', ')}</td>
                </tr>
                <tr key="Normalizer">
                    <td>Normalizer</td>
                    <td>{optimizer.NormalizerDataSelector}</td>
                    <td>{optimizer.NormalizerDataSelectorParameters}</td>
                    <td>{datasets.Normalizer.join(', ')}</td>
                </tr>
            </tbody>
        </Table>
    );
}

export default DataTable;
import React from "react";
import {
  Table,
} from "reactstrap";
// import { Table } from 'react-bootstrap';

// function OptTable({ optimizer }) {
//     return (
//         <Table lg={12} md={12} sm={12} striped>
//             <thead>
//                 <tr className="fs-sm">
//                     <th>#</th>
//                     <th>Narrow Search Space</th>
//                     <th>Initialization</th>
//                     <th>Pre-train</th>
//                     <th>Surrogate Model</th>
//                     <th>Acquisition Function</th>
//                     <th>Normalizer</th>
//                 </tr>
//             </thead>
//             <tbody>
//                 <tr key="Name">
//                     <td>Name</td>
//                     <td>{optimizer.SpaceRefiner}</td>
//                     <td>{optimizer.Sampler}</td>
//                     <td>{optimizer.Pretrain}</td>
//                     <td>{optimizer.Model}</td>
//                     <td>{optimizer.ACF}</td>
//                     <td>{optimizer.Normalizer}</td>
//                 </tr>
//                 <tr key="Parameters">
//                     <td>Parameters</td>
//                     <td>{optimizer.SpaceRefinerParameters}</td>
//                     <td>InitNum:{optimizer.SamplerInitNum},{optimizer.SamplerParameters}</td>
//                     <td>{optimizer.PretrainParameters}</td>
//                     <td>{optimizer.ModelParameters}</td>
//                     <td>{optimizer.ACFParameters}</td>
//                     <td>{optimizer.NormalizerParameters}</td>
//                 </tr>
//             </tbody>
//         </Table>
//     );
// }



function OptTable({ optimizer }) {
    return (
        <Table lg={12} md={12} sm={12} striped>
            <thead>
                <tr className="fs-sm">
                    <th>#</th>
                    <th>Name</th>
                    <th>Parameters</th>
                </tr>
            </thead>
            <tbody>
                <tr key="Name">
                    <td>Prune Search Space</td>
                    <td>{optimizer.SpaceRefiner}</td>
                    <td>{optimizer.SpaceRefinerParameters}</td>
                </tr>

                <tr key="Name">
                    <td>Initialization</td>
                    <td>{optimizer.Sampler}</td>
                    <td>The number of Initialization:{optimizer.SamplerInitNum},{optimizer.SamplerParameters}</td>
                </tr>

                <tr key="Name">
                    <td>Pre-train</td>
                    <td>{optimizer.Pretrain}</td>
                    <td>{optimizer.PretrainParameters}</td>

                </tr>
                <tr key="Name">
                    <td>Surrogate Model</td>
                    <td>{optimizer.Model}</td>
                    <td>{optimizer.ModelParameters}</td>

                </tr>
                <tr key="Name">
                    <td>Acquisition Function</td>
                    <td>{optimizer.ACF}</td>
                    <td>{optimizer.ACFParameters}</td>
                </tr>
                
                <tr key="Name">
                    <td>Normalizer</td>
                    <td>{optimizer.Normalizer}</td>
                    <td>{optimizer.NormalizerParameters}</td>

                </tr>

            </tbody>
        </Table>
    );
}


export default OptTable;


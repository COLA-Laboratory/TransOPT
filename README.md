<p align="center">
  <a href="https://maopl.github.io/TransOpt-doc/">
    <img src="./docs/source/_static/figures/transopt_logo.jpg" alt="" width="40%" align="top">
  </a>
</p>
<p align="center">
  TransOPT: Transfer Optimization System for Bayesian Optimization Using Transfer Learning<br>
  <a href="https://leopard-ai.github.io/betty/">Docs</a> |
  <a href="https://leopard-ai.github.io/betty/tutorial/basic/basic.html">Tutorials</a> |
  <a href="https://github.com/leopard-ai/betty/tree/main/examples">Examples</a> |
  <a href="https://openreview.net/pdf?id=LV_MeMS38Q9">Paper</a> |
  <a href="https://github.com/leopard-ai/betty#citation">Citation</a> |
</p>

<div align="center">

  <a href="https://pypi.org/project/betty-ml/">![Version](https://img.shields.io/pypi/v/betty-ml)</a>
  <a href="https://github.com/leopard-ai/betty/tree/main/test">![Testing](https://img.shields.io/github/actions/workflow/status/leopard-ai/betty/test.yaml?branch=main)</a>
  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/leopard-ai/betty/blob/main/LICENSE)
  <a href="https://arxiv.org/abs/2207.02849">![arXiv](https://img.shields.io/badge/arXiv-2207.02489-b31b1b.svg)</a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="code style: black"></a>
  <a href="https://join.slack.com/t/betty-n2l2441/shared_invite/zt-1ojhxizmt-NTmj2aVi3BuQQ6hjhNBTFQ" target="_blank">
    <img alt="Slack" src="https://img.shields.io/badge/Slack-Join%20Slack-blueviolet?logo=slack" />
  </a>
  
</div>


# Introduction

**TransOPT** is an open-source software platform designed to facilitate the **design, benchmarking, and application of transfer learning for Bayesian optimization (TLBO)** algorithms through a modular, data-centric framework.

## Features

- **Access a variety of benchmark problems** with ease to evaluate and compare algorithms.  
- **Build custom optimization algorithms** as easily as stacking building blocks.  
- **Leverage historical data** to achieve more efficient and informed optimization.  
- **Deploy experiments through an intuitive UI** and **monitor results in real-time**.

TransOPT empowers researchers and developers to explore innovative optimization solutions effortlessly, bridging the gap between theory and practical application.

## Installation

TransOPT is composed of two main components: the backend for data processing and business logic, and the frontend for user interaction. Each can be installed as follows:

### Prerequisites

Before installing TransOPT, you must have the following installed:

- **Python 3.8+**: Ensure Python is installed.
- **Node.js and npm**: These are required to install and build the frontend. [Download Node.js](https://nodejs.org/en/download/)

Please install these prerequisites if they are not already installed on your system.

### Backend Installation

**Install from source:**
   ```shell
   $ git clone --recurse-submodules https://github.com/maopl/TransOPT.git 
   $ cd TransOPT
   $ pip install -r requirements.txt
   $ python setup.py install
   $ bash scripts/init_docker.sh
   $ bash scripts/init_csstuning.sh
   ```

### Frontend Installation

Navigate to the `webui` directory and install the necessary packages:

```shell
$ cd webui && npm install
```

## Getting Started

After installation, you can start the backend and frontend as follows:

1. Start the backend server:
   ```shell
   $ python transopt/agent/app.py
   ```

2. In a new terminal, launch the frontend:
   ```shell
   $ cd webui && npm start
   ```
This will open the TransOPT interface in your default web browser at `http://localhost:3000`.

## Documentation

For more detailed information on configuring and using TransOPT, refer to our full documentation [here](https://maopl.github.io/TransOpt-doc/).

## Support

For issues, feature requests, or contributions, please visit our [Documentation](https://maopl.github.io/TransOpt-doc/) page.


## Citation

If you find our work helpful to your research, please consider citing our:

```bibtex
@software{TransOPT,
  title = {{TransOPT}: Tansfer Optimization System For Automated Configuration},
  author = {Author Name and Collaborator Name},
  url = {https://github.com/maopl/TransOPT},
  year = {2024}
}
```


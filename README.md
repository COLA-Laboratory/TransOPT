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

  <a href="https://github.com/leopard-ai/betty/tree/main/test">![Testing](https://img.shields.io/github/actions/workflow/status/leopard-ai/betty/test.yaml?branch=main)</a>
  [![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/peilimao/TransOpt/blob/main/LICENSE)
  <a href="https://arxiv.org/abs/2207.02849">![arXiv](https://img.shields.io/badge/arXiv-2207.02489-b31b1b.svg)</a>
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

- **Python 3.10+**: Ensure Python is installed.
- **Node.js 17.9.1+ and npm 8.11.0+**: These are required to install and build the frontend. [Download Node.js](https://nodejs.org/en/download/)

Please install these prerequisites if they are not already installed on your system.

1. Clone the repository:
   ```shell
   $ git clone https://github.com/maopl/TransOpt.git
   ```

2. Install the required dependencies:
   ```shell
   $ cd TransOpt
   $ python setup.py install
   ```

3. Install the frontend dependencies:
   ```shell
   $ cd webui && npm install
   ```



## Quick Start

### Start the Backend Agent

To start the backend agent, use the following command:

```bash
$ python transopt/agent/app.py
```

### Web User Interface Mode

When TransOPT has been started successfully, go to the webui directory and start the web UI on your local machine. Enable the user interface mode with the following command:
```bash
cd webui && npm start
```

This will open the TransOPT interface in your default web browser at `http://localhost:3000`.


### Command Line Mode

In addition to the web UI mode, TransOPT also offers a Command Line (CMD) mode for users who may not have access to a display screen, such as when working on a remote server.

To run TransOPT in CMD mode, use the following command:

```bash
python transopt/agent/run_cli.py -n MyTask -v 3 -o 2 -m RF -acf UCB -b 300
```

This command sets up a task named MyTask with 3 variables and 2 objectives, using a Random Forest model (RF) and the Upper Confidence Bound (UCB) acquisition function, with a budget of 300 function evaluations.

For a complete list of available options and more detailed usage instructions, please refer to the [CLI documentation](https://maopl.github.io/TransOpt-doc/usage/cli.html).


## Benchmark Problem

### Synthetic Problem

TransOPT contains more than several synthetic benchmark problems. Here we demonstrate the usage with the Ackley function, which is a widely used benchmark function in optimization.

The Ackley function looks like this:

![Ackley Function](docs/source/_static/figures/visualization/Rastrigin.png)

Here's a simple example of how to use TransOPT to optimize the Ackley function:

### Hyperparameter Optimization Problem

### Configurable Software System Tuning

### RNA Inverse Design

### Protein Inverse Design


## Documentation

For more detailed information on configuring and using TransOPT, refer to our full documentation [here](https://maopl.github.io/TransOpt-doc/).

## Support

For issues, feature requests, or contributions, please visit our [Documentation](https://maopl.github.io/TransOpt-doc/) page.


## Citation

If you find our work helpful to your research, please consider citing our:

```bibtex
@article{TransOPT,
  title = {{TransOPT}: Transfer Optimization System for Bayesian Optimization Using Transfer Learning},
  author = {Author Name and Collaborator Name},
  url = {https://github.com/maopl/TransOPT},
  year = {2024}
}
```




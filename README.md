<img src="" alt="TransOpt logo" width="200" height="200" />

**TranOpt:**

## Overview

TransOPT is a sophisticated web-based platform designed to facilitate optimization experiments ...

## Features

...

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

For more detailed information on configuring and using TransOPT, refer to our full documentation [here](link-to-documentation).

## Support

For issues, feature requests, or contributions, please visit our [GitHub Issues](link-to-issues) page.


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


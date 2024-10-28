Installation Guide
==================

This section will guide you through the steps required to install TransOPT on your system.

Before installing, ensure you have the following installed:

- Python 3.10
- Node.js 17.9.1
- npm 8.11.0

1. Clone the repository:

   .. code-block:: console 

     $ git clone https://github.com/maopl/TransOpt.git

2. Install the required dependencies:

   .. code-block:: console 

     $ cd TransOpt
     $ python setup.py install


3. Install the frontend dependencies:

   .. code-block:: console 

     $ cd webui && npm install

4. (Optional) Install additional extensions:

   You can enhance the functionality of the system by installing the following optional packages:

   - **Extension 1**: Provides advanced results analysis.

     .. code-block:: console 

       $ python setup.py install[analysis]

   - **Extension 2**: Adds support for distributed computing.

     .. code-block:: console 

       $ python setup.py install[remote]


5. (Optional) Install optional Docker containers:

   The following Docker containers are available to provide additional problem generators:

   - **Inverse RNA Design**: Provides inverse RNA design problem generators:

     .. code-block:: console 

       $ bash scripts/init_docker.sh

   - **Protein Design**: Adds support for distributed computing.

     .. code-block:: console 

       $ bash scripts/init_csstuning.sh

   - **Configurable Software Tuning**: Enables integration with external APIs.

     .. code-block:: console 

       $ bash scripts/init_csstuning.sh
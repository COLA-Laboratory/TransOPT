Quick Start
======================

TransOPT is a sophisticated system designed to facilitate transfer optimization services and experiments. It is composed of two parts: the agent and the web user interface. The agent is responsible for running the optimization algorithms and the web user interface is responsible for displaying the results and managing the experiments.

Start the backend agent:

.. code-block:: console 

  $ python transopt/agent/app.py



Web User Interface Mode
-----------------------
When TransOPT has been started successfully, go to the directory of webui and start the webui on your local machine. Enable the user interface mode with the following command:

.. code-block:: console 

  $ cd webui && npm start



Command Line Mode
-----------------

In addition to the web UI mode, TransOPT also offers a Command Line (CMD) mode for users who may not have access to a display screen, such as when working on a remote server.

To run TransOPT in CMD mode, use the following command:

.. code-block:: console 

  $ python transopt/agent/run_cli.py -n MyTask -v 3 -o 2 -m RF -acf UCB -b 300

This command sets up a task named `MyTask` with 3 variables and 2 objectives, using a Random Forest model (`RF`) and the Upper Confidence Bound (`UCB`) acquisition function, with a budget of 300 function evaluations.

For a complete list of available options and more detailed usage instructions, please refer to the :ref:`CLI documentation <command_line_usage>`.

.. _command_line_usage:

Command Line
===============================

TransOPT provides a command-line interface (CLI) that allows users to define and run optimization tasks directly from the terminal. This is facilitated by the `run_cli.py` script, which supports a wide range of customizable parameters.

Running the Command-Line Interface
----------------------------------

To run the `run_cli.py` script, navigate to the directory containing the script and use the following command:

.. code-block:: bash

   python transopt/agent/run_cli.py [OPTIONS]

Where `[OPTIONS]` are the command-line arguments you can specify to customize the behavior of TransOPT.

I. Command-Line Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^
Here is a list of the main command-line arguments supported by the script:

**Task Configuration**

- **`-n, --task_name`**: Name of the task (default: `"Sphere"`).
- **`-v, --num_vars`**: Number of variables (default: `2`).
- **`-o, --num_objs`**: Number of objectives (default: `1`).
- **`-f, --fidelity`**: Fidelity level of the task (default: `""`).
- **`-w, --workloads`**: Workloads associated with the task (default: `"0"`).
- **`-bt, --budget_type`**: Type of budget (e.g., `"Num_FEs"`) (default: `"Num_FEs"`).
- **`-b, --budget`**: Budget for the task, typically the number of function evaluations (default: `100`).

**Optimizer Configuration**

- **`-sr, --space_refiner`**: Space refiner method (default: `"None"`).
- **`-srp, --space_refiner_parameters`**: Parameters for the space refiner (default: `""`).
- **`-srd, --space_refiner_data_selector`**: Data selector for the space refiner (default: `"None"`).
- **`-srdp, --space_refiner_data_selector_parameters`**: Parameters for the data selector (default: `""`).
- **`-sp, --sampler`**: Sampling method (default: `"random"`).
- **`-spi, --sampler_init_num`**: Initial number of samples (default: `22`).
- **`-spp, --sampler_parameters`**: Parameters for the sampler (default: `""`).
- **`-spd, --sampler_data_selector`**: Data selector for the sampler (default: `"None"`).
- **`-spdp, --sampler_data_selector_parameters`**: Parameters for the sampler's data selector (default: `""`).
- **`-pt, --pre_train`**: Pretraining method (default: `"None"`).
- **`-ptp, --pre_train_parameters`**: Parameters for pretraining (default: `""`).
- **`-ptd, --pre_train_data_selector`**: Data selector for pretraining (default: `"None"`).
- **`-ptdp, --pre_train_data_selector_parameters`**: Parameters for the pretraining data selector (default: `""`).
- **`-m, --model`**: Model used for optimization (default: `"GP"`).
- **`-mp, --model_parameters`**: Parameters for the model (default: `""`).
- **`-md, --model_data_selector`**: Data selector for the model (default: `"None"`).
- **`-mdp, --model_data_selector_parameters`**: Parameters for the model's data selector (default: `""`).
- **`-acf, --acquisition_function`**: Acquisition function used (default: `"EI"`).
- **`-acfp, --acquisition_function_parameters`**: Parameters for the acquisition function (default: `""`).
- **`-acfd, --acquisition_function_data_selector`**: Data selector for the acquisition function (default: `"None"`).
- **`-acfdp, --acquisition_function_data_selector_parameters`**: Parameters for the acquisition function's data selector (default: `""`).
- **`-norm, --normalizer`**: Normalization method (default: `"Standard"`).
- **`-normp, --normalizer_parameters`**: Parameters for the normalizer (default: `""`).
- **`-normd, --normalizer_data_selector`**: Data selector for the normalizer (default: `"None"`).
- **`-normdp, --normalizer_data_selector_parameters`**: Parameters for the normalizer's data selector (default: `""`).

**General Configuration**

- **`-s, --seeds`**: Random seed for reproducibility (default: `0`).

II. Example Usage
^^^^^^^^^^^^^^^^^
Below are some example commands demonstrating how to use the CLI to run different tasks with varying configurations.

**Example 1: Running a basic task with default parameters**

.. code-block:: bash

   python transopt/agent/run_cli.py -n MyTask -v 3 -o 1 -b 200

**Example 2: Running a task with a specific model and acquisition function**

.. code-block:: bash

   python transopt/agent/run_cli.py -n MyTask -v 3 -o 2 -m RF -acf UCB -b 300

**Example 3: Using custom parameters for the space refiner and sampler**

.. code-block:: bash

   python transopt/agent/run_cli.py -n MyTask -sr "Prune"  -sp "lhs" -spi 30 -b 300

III. Additional Notes
^^^^^^^^^^^^^^^^^^^^^
- The **random seed** is particularly important for ensuring that the results are reproducible. Make sure to specify the `--seeds` option if you want to run experiments that can be exactly replicated.
- TransOPT's CLI is highly flexible, allowing you to tailor the optimization process to your specific needs by adjusting the parameters and options provided.

By following the instructions above, you can effectively use the TransOPT CLI to run and manage your optimization tasks.

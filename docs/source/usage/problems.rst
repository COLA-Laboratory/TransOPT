Benchmark Problems
==================
This

.. admonition:: Overview
   :class: info

   - :ref:`Register <registering-new-problem>`: How to register a new optimization problem to :ref:`TransOPT <home>`
   - :ref:`Synthetic Problem <synthetic-problems>`: The list of the synthetic problems available in :ref:`TransOPT <home>`
   - :ref:`Hyperparameter Optimization Problem <hpo-problems>`: The list of the HPO problems available in :ref:`TransOPT <home>`
   - :ref:`Configurable Software Optimization Problem <cso-problems>`: The list of the configurable software optimization problems available in :ref:`TransOPT <home>`
   - :ref:`RNA Inverse Design Problem <rna-problems>`: The list of the RNA Inverse design problems available in :ref:`TransOPT <home>`
   - :ref:`Protein Inverse Folding Problem <pif-problems>`: The list of the protein inverse folding problems available in :ref:`TransOPT <home>`
   - :ref:`Parallelization <parallelization>`: How to parallelize function evaluations


.. _registering-new-problem:


Registering a New Benchmark Problem
-----------------------------------

To register a new benchmark problem in the TransOPT framework, follow the steps below.

I. Import the Problem Registry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First, you need to import the `problem_registry` from the `transopt.agent.registry` module:

.. code-block:: python

    from transopt.agent.registry import problem_registry

II. Define a New Problem Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Next, define a new problem class. This class should be decorated with the `@problem_registry.register("ProblemName")` decorator, where `"ProblemName"` is the unique identifier for the problem. The new problem class must inherit from one of the following base classes:

- `NonTabularProblem`
- `TabularProblem`

For example, to create a new problem named "new_problem", you would define the class as follows:

.. code-block:: python

    @problem_registry.register("new_problem")
    class new_problem(NonTabularProblem):
        pass  # Further implementation required

III. Implement Required Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After defining the class, you need to implement the following three abstract methods:

1. **get_configuration_space**: 
   This method is responsible for defining the configuration space of the new problem.

   .. code-block:: python

       def get_configuration_space(self):
           # Define and return the configuration space
           pass

2. **get_fidelity_space**: 
   This method should define the fidelity space for the problem, if applicable.

   .. code-block:: python

       def get_fidelity_space(self):
           # Define and return the fidelity space
           pass

3. **objective_function**: 
   This method evaluates the problem's objective function based on the provided configuration and other parameters.

   .. code-block:: python

       def objective_function(self, configuration, fidelity=None, seed=None, **kwargs) -> Dict:
           # Evaluate the configuration and return the results as a dictionary
           pass

Here’s an example outline of the `sphere` class:

.. code-block:: python

    @problem_registry.register("sphere")
    class sphere(NonTabularProblem):
        
      def get_configuration_space(self):
            # Define the configuration space here
         variables =  [Continuous(f'x{i}', (-5.12, 5.12)) for i in range(self.input_dim)]
         ss = SearchSpace(variables)
         return ss
        
      def get_fidelity_space(self) -> FidelitySpace:
         fs = FidelitySpace([])
         return fs

      def objective_function(self, configuration, fidelity=None, seed=None, **kwargs) -> Dict:
         # Implement the evaluation logic and return the results as a dictionary
         X = np.array([[configuration[k] for idx, k in enumerate(configuration.keys())]])
         y = np.sum((X) ** 2, axis=1)
         results = {'function_value': float(y)}

         return results

By following these steps, you can successfully register a new benchmark problem in the TransOPT framework.

.. _synthetic-problems:

Synthetic Problem
------------------

The synthetic problems in this section are widely used in the optimization literature for benchmarking optimization algorithms. These problems exhibit diverse characteristics and levels of complexity, making them ideal for testing the robustness and efficiency of different optimization strategies. Below is an overview of the synthetic problems included in this benchmark suite:

- **Sphere:** A simple convex problem that is often used as a baseline. The global minimum is located at the origin, and the objective function value increases quadratically with distance from the origin.

- **Rastrigin:** A non-convex problem characterized by a large number of local minima, making it challenging for optimization algorithms to find the global minimum.

- **Schwefel:** Known for its complex landscape with many local minima, the Schwefel function requires optimization algorithms to balance exploration and exploitation effectively.

- **Ackley:** A multi-modal function with a nearly flat outer region and a large hole at the center, making it difficult for algorithms to escape local minima and converge to the global minimum.

- **Levy:** A multi-modal problem with a complex landscape that tests an algorithm's ability to handle irregularities and identify global optima.

- **Griewank:** A function with many widespread local minima, making it challenging to converge to the global optimum. It is often used to assess the ability of algorithms to avoid getting trapped in local minima.

- **Rosenbrock:** A non-convex problem with a narrow, curved valley that contains the global minimum. This function is commonly used to test the convergence properties of optimization algorithms.

- **Dropwave:** A challenging multi-modal function with steep drops, requiring careful search strategies to avoid local minima.

- **Langermann:** This problem has many local minima and a highly irregular structure, testing an algorithm's ability to explore complex search spaces.

- **Rotated Hyper-Ellipsoid:** A rotated version of the ellipsoid function, which tests an algorithm's capability to optimize problems with rotated and ill-conditioned landscapes.

- **Sum of Different Powers:** A problem where each term in the sum contributes differently to the overall objective, requiring optimization algorithms to handle varying sensitivities across dimensions.

- **Styblinski-Tang:** A function with multiple global minima, commonly used to test an algorithm's ability to avoid suboptimal solutions.

- **Powell:** A problem designed to challenge optimization algorithms with a mixture of convex and non-convex characteristics across different dimensions.

- **Dixon-Price:** This function has a smooth, narrow valley leading to the global minimum, testing an algorithm’s ability to navigate such features.

- **Ellipsoid:** A test problem that features high conditioning and elliptical level sets, requiring algorithms to efficiently search in skewed spaces.

- **Discus:** A variant of the sphere function with a large difference in scale between the first variable and the rest, making it a test of handling unbalanced scales.

- **BentCigar:** A highly anisotropic function where one direction has a much larger scale than the others, challenging algorithms to adjust their search strategies accordingly.

- **SharpRidge:** This function has a sharp ridge along one dimension, testing an algorithm's ability to optimize in narrow, high-gradient regions.

- **Katsuura:** A multi-fractal function that combines periodicity and complexity, testing the capability of algorithms to explore intricate landscapes.

- **Weierstrass:** A problem with a fractal structure, characterized by a large number of local minima and requiring algorithms to handle varying scales of roughness.

- **Different Powers:** A problem where each term contributes differently to the objective, challenging algorithms to manage varying sensitivities and scales.

- **Trid:** A function that has a curved and ridge-like structure, often used to assess the convergence properties of optimization algorithms.

- **LinearSlope:** A simple linear function with a varying slope across dimensions, used to test the basic exploration capabilities of optimization methods.

- **Elliptic:** Similar to the Ellipsoid function but with exponentially increasing scales, testing an algorithm’s ability to search efficiently in poorly conditioned spaces.

- **PERM:** A complex combinatorial problem that combines different power terms, testing an algorithm’s ability to handle permutation-based search spaces.

- **Power Sum:** A problem where each dimension contributes a power sum to the objective, requiring algorithms to handle large variations in sensitivity across variables.

- **Zakharov:** A problem with a complex, non-linear interaction between variables, used to test an algorithm’s ability to navigate multi-variable coupling.

- **Six-Hump Camel:** A low-dimensional, multi-modal problem with several local minima, requiring precise search strategies to find the global optimum.

- **Michalewicz:** A problem known for its challenging steepness and periodicity, making it difficult for algorithms to locate the global minimum.

- **Moving Peak:** A dynamic optimization problem where the objective function changes over time, used to assess an algorithm’s adaptability to changing landscapes.

These problems collectively provide a comprehensive suite for evaluating optimization algorithms across a broad range of difficulties, including convexity, multi-modality, separability, and conditioning.

+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
|      Problem name       |                                                                       Mathematical formulation                                                                        |              Range                       |                               |                             |
+=========================+=======================================================================================================================================================================+==========================================+===============================+=============================+
| Sphere                  | :math:`f(\mathbf{x}) = \sum_{i=1}^d x_i^2`                                                                                                                            | :math:`x_i \in [-5.12, 5.12]`            |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Rastrigin               | :math:`f(\mathbf{x}) = 10 d + \sum_{i=1}^d \left[ x_i^2 - 10 \cos(2 \pi x_i) \right]`                                                                                 | :math:`x_i \in [-32.768, 32.768]`        |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Schwefel                | :math:`f(\mathbf{x}) = 418.9829 d - \sum_{i=1}^d x_i \sin\left(\sqrt{\left|x_i\right|}\right)`                                                                        | :math:`x_i \in [-500, 500]`              |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Ackley                  | :math:`f(\mathbf{x}) = -a \exp \left(-b \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2}\right)`                                                                                 | :math:`x_i \in [-32.768, 32.768]`        |                               |                             |
|                         | :math:`-\exp \left(\frac{1}{d} \sum_{i=1}^d \cos \left(c x_i\right)\right) + a + \exp(1)`                                                                             |                                          |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Levy                    | :math:`f(\mathbf{x}) = \sin^2\left(\pi w_1\right) + \sum_{i=1}^{d-1}\left(w_i - 1\right)^2`                                                                           | :math:`x_i \in [-10, 10]`                |                               |                             |
|                         | :math:`\left[1 + 10 \sin^2\left(\pi w_i + 1\right)\right] + \left(w_d - 1\right)^2`                                                                                   |                                          |                               |                             |
|                         | :math:`\left[1 + \sin^2\left(2 \pi w_d\right)\right], w_i = 1 + \frac{x_i - 1}{4}`                                                                                    |                                          |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Griewank                | :math:`f(\mathbf{x}) = \sum_{i=1}^d \frac{x_i^2}{4000} - \prod_{i=1}^d \cos\left(\frac{x_i}{\sqrt{i}}\right) + 1`                                                     | :math:`x_i \in [-600, 600]`              |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Rosenbrock              | :math:`f(\mathbf{x}) = \sum_{i=1}^{d-1}\left[100\left(x_{i+1} - x_i^2\right)^2 + \left(x_i - 1\right)^2\right]`                                                       | :math:`x_i \in [-5, 10]`                 |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Dropwave                | :math:`f(\mathbf{x}) = -\frac{1 + \cos\left(12 \sqrt{x_1^2 + x_2^2}\right)}{0.5\left(x_1^2 + x_2^2\right) + 2}`                                                       | :math:`x_i \in [-5.12, 5.12]`            |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Langermann              | :math:`f(\mathbf{x}) = \sum_{i=1}^m c_i \exp\left(-\frac{1}{\pi} \sum_{j=1}^d \left(x_j - A_{ij}\right)^2\right)`                                                     | :math:`x_i \in [0, 10]`                  |                               |                             |
|                         | :math:`\cos\left(\pi \sum_{j=1}^d\left(x_j - A_{ij}\right)^2\right)`                                                                                                  |                                          |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Rotated Hyper-Ellipsoid | :math:`f(\mathbf{x}) = \sum_{i=1}^d \sum_{j=1}^i x_j^2`                                                                                                               | :math:`x_i \in [-65.536, 65.536]`        |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Sum of Different Powers | :math:`f(\mathbf{x}) = \sum_{i=1}^d x_i^{i+1}`                                                                                                                        | :math:`x_i \in [-1, 1]`                  |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Styblinski-Tang         | :math:`f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^d\left(x_i^4 - 16 x_i^2 + 5 x_i\right)`                                                                                 | :math:`x_i \in [-5, 5]`                  |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Powell                  | :math:`f(\mathbf{x}) = \sum_{i=1}^{d/4}\left(x_{4i-3} + 10 x_{4i-2}\right)^2`                                                                                         | :math:`x_i \in [-4, 5]`                  |                               |                             |
|                         | :math:`+ 5\left(x_{4i-1} - x_{4i}\right)^2`                                                                                                                           |                                          |                               |                             |
|                         | :math:`+ \left(x_{4i-2} - 2 x_{4i-1}\right)^4`                                                                                                                        |                                          |                               |                             |
|                         | :math:`+ 10\left(x_{4i-3} - x_{4i}\right)^4`                                                                                                                          |                                          |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Dixon-Price             | :math:`f(\mathbf{x}) = \left(x_1 - 1\right)^2 + \sum_{i=2}^d i\left(2 x_i^2 - x_{i-1}\right)^2`                                                                       | :math:`x_i \in [-10, 10]`                |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Ellipsoid               | :math:`f_2(\mathbf{x}) = \sum_{i=1}^D 10^{6 \frac{i-1}{D-1}} z_i^2 + f_{\mathrm{opt}}`                                                                                | :math:`x_i \in [-5, 5]`                  |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Discus                  | :math:`f(\mathbf{x}) = 10^6 x_1^2 + \sum_{i=2}^D x_i^2`                                                                                                               | :math:`x_i \in [-5, 5]`                  |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| BentCigar               | :math:`f(\mathbf{x}) = x_1^2 + 10^6 \sum_{i=2}^n x_i^2`                                                                                                               | :math:`x_i \in [-5, 5]`                  |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| SharpRidge              | :math:`f(\mathbf{x}) = x_1^2 + 100 \sqrt{\sum_{i=2}^D x_i^2}`                                                                                                         | :math:`x_i \in [-5, 5]`                  |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Katsuura                | :math:`f(\mathbf{x}) = \frac{10}{D^2} \prod_{i=1}^D \left(1 + i \sum_{j=1}^{32} \frac{2^j x_i - \left[2^j x_i\right]}{2^j}\right)^{10 / D^{1.2}}`                     | :math:`x_i \in [-5, 5]`                  |                               |                             |
|                         | :math:`- \frac{10}{D^2} + f_{\mathrm{pen}}(\mathbf{x})`                                                                                                               |                                          |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Weierstrass             | :math:`f_{16}(\mathbf{x}) = 10 \left(\frac{1}{D} \sum_{i=1}^D \sum_{k=0}^{11} \frac{1}{2^k} \cos \left(2 \pi 3^k\left(z_i + \frac{1}{2}\right)\right) - f_0\right)^3` | :math:`x_i \in [-5, 5]`                  |                               |                             |
|                         | :math:`+ \frac{10}{D} f_{\mathrm{pen}}(\mathbf{x})`                                                                                                                   |                                          |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| DifferentPowers         | :math:`f(\mathbf{x}) = \sqrt{\sum_{i=1}^D x_i^{2 + 4 \frac{i-1}{D-1}}}`                                                                                               | :math:`x_i \in [-5, 5]`                  |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Trid                    | :math:`f(\mathbf{x}) = \sum_{i=1}^d \left(x_i - 1\right)^2 - \sum_{i=2}^d x_i x_{i-1}`                                                                                | :math:`x_i \in [-d^2, d^2]`              |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| LinearSlope             | :math:`f(\mathbf{x}) = \sum_{i=1}^D 5 s_i - s_i x_i`                                                                                                                  | :math:`x_i \in [-5, 5]`                  |                               |                             |
|                         | :math:`s_i = \operatorname{sign}\left(x_i^{\mathrm{opt}}\right) 10^{\frac{i-1}{D-1}},`                                                                                |                                          |                               |                             |
|                         | :math:`\text{for } i=1, \ldots, D`                                                                                                                                    |                                          |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Elliptic                | :math:`f(\mathbf{x}) = \sum_{i=1}^D \left(10^6\right)^{\frac{i-1}{D-1}} x_i^2`                                                                                        | :math:`x_i \in [-5, 5]`                  |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| PERM                    | :math:`f(\mathbf{x}) = \sum_{i=1}^d \left(\sum_{j=1}^d \left(j + \beta\right)\left(x_j^i - \frac{1}{j^i}\right)\right)^2`                                             | :math:`x_i \in [-d, d]`                  |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Power Sum               | :math:`f(\mathbf{x}) = \sum_{i=1}^d \left[\left(\sum_{j=1}^d x_j^i\right) - b_i\right]^2`                                                                             | :math:`x_i \in [0, d]`                   |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Zakharov                | :math:`f(\mathbf{x}) = \sum_{i=1}^d x_i^2 + \left(\sum_{i=1}^d 0.5 i x_i\right)^2`                                                                                    | :math:`x_i \in [-5, 10]`                 |                               |                             |
|                         | :math:`+ \left(\sum_{i=1}^d 0.5 i x_i\right)^4`                                                                                                                       |                                          |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Six-Hump Camel          | :math:`f(\mathbf{x}) = \left(4 - 2.1 x_1^2 + \frac{x_1^4}{3}\right) x_1^2 + x_1 x_2`                                                                                  | :math:`x_1 \in [-3, 3], x_2 \in [-2, 2]` |                               |                             |
|                         | :math:`+ \left(-4 + 4 x_2^2\right) x_2^2`                                                                                                                             |                                          |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Michalewicz             | :math:`f(\mathbf{x}) = -\sum_{i=1}^d \sin \left(x_i\right) \sin ^{2 m}\left(\frac{i x_i^2}{\pi}\right)`                                                               | :math:`x_i \in [0, \pi]`                 |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| Moving Peak             | :math:`f(\mathbf{x}) = \sum_{i=1}^D \left(10^6\right)^{\frac{i-1}{D-1}} x_i^2`                                                                                        | :math:`x_i \in [0, 100]`                 |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+
| PERM 2                  | :math:`f(\mathbf{x}) = \sum_{i=1}^d\left(\sum_{j=1}^d\left(j^i+\beta\right)\left(\left(\frac{x_j}{j}\right)^i-1\right)\right)^2`                                      | :math:`x_i \in [-d, d]`                  |                               |                             |
+-------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------+------------------------------------------+-------------------------------+-----------------------------+


.. _hpo-problems:

Hyperparameter Optimization Problem
------------------------------------

This section provides an overview of the hyperparameter optimization problem including the hyperparameters used for various machine learning models and machine learning tasks used for generate problem instances.

Hyperparameters for Support Vector Machine (SVM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Support Vector Machines (SVM) are widely used for classification and regression tasks. They are particularly effective in high-dimensional spaces and situations where the number of dimensions exceeds the number of samples. The hyperparameters for SVM control the regularization and the kernel function, which are crucial for model performance.

+--------------------+-------------------+------------+
| **Hyperparameter** |     **Range**     |  **Type**  |
+====================+===================+============+
| C                  | :math:`[-10, 10]` | Continuous |
+--------------------+-------------------+------------+
| gamma              | :math:`[-10, 10]` | Continuous |
+--------------------+-------------------+------------+

Hyperparameters for AdaBoost
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AdaBoost is a popular ensemble method that combines multiple weak learners to create a strong classifier. It is particularly useful for boosting the performance of decision trees. The hyperparameters control the number of estimators and the learning rate, which affects the contribution of each classifier.

+--------------------+-------------------+------------+
| **Hyperparameter** |     **Range**     |  **Type**  |
+====================+===================+============+
| n_estimators       | :math:`[1, 100]`  | Integer    |
+--------------------+-------------------+------------+
| learning_rate      | :math:`[0.01, 1]` | Continuous |
+--------------------+-------------------+------------+

Hyperparameters for Random Forest
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Random Forest is an ensemble learning method that builds multiple decision trees and merges them to get a more accurate and stable prediction. It is widely used for both classification and regression tasks. The hyperparameters include the number of trees, the depth of the trees, and various criteria for splitting nodes.

+--------------------------+--------------------+-------------+
|    **Hyperparameter**    |     **Range**      |  **Type**   |
+==========================+====================+=============+
| n_estimators             | :math:`[1, 1000]`  | Integer     |
+--------------------------+--------------------+-------------+
| max_depth                | :math:`[1, 100]`   | Integer     |
+--------------------------+--------------------+-------------+
| criterion                | {gini, entropy}    | Categorical |
+--------------------------+--------------------+-------------+
| min_samples_leaf         | :math:`[1, 20]`    | Integer     |
+--------------------------+--------------------+-------------+
| min_weight_fraction_leaf | :math:`[0.0, 0.5]` | Continuous  |
+--------------------------+--------------------+-------------+
| min_impurity_decrease    | :math:`[0.0, 1.0]` | Continuous  |
+--------------------------+--------------------+-------------+

Hyperparameters for XGBoost
^^^^^^^^^^^^^^^^^^^^^^^^^^^

XGBoost is an efficient and scalable implementation of gradient boosting, designed for speed and performance. It is widely used in machine learning competitions and industry for classification and regression tasks. The hyperparameters include learning rates, tree depths, and regularization parameters, which control the complexity of the model and its ability to generalize.

+--------------------+-----------------------+------------+
| **Hyperparameter** |       **Range**       |  **Type**  |
+====================+=======================+============+
| eta                | :math:`[-10.0, 0.0]`  | Continuous |
+--------------------+-----------------------+------------+
| max_depth          | :math:`[1, 15]`       | Integer    |
+--------------------+-----------------------+------------+
| min_child_weight   | :math:`[0.0, 7.0]`    | Continuous |
+--------------------+-----------------------+------------+
| colsample_bytree   | :math:`[0.01, 1.0]`   | Continuous |
+--------------------+-----------------------+------------+
| colsample_bylevel  | :math:`[0.01, 1.0]`   | Continuous |
+--------------------+-----------------------+------------+
| reg_lambda         | :math:`[-10.0, 10.0]` | Continuous |
+--------------------+-----------------------+------------+
| reg_alpha          | :math:`[-10.0, 10.0]` | Continuous |
+--------------------+-----------------------+------------+
| subsample_per_it   | :math:`[0.1, 1.0]`    | Continuous |
+--------------------+-----------------------+------------+
| n_estimators       | :math:`[1, 50]`       | Integer    |
+--------------------+-----------------------+------------+
| gamma              | :math:`[0.0, 1.0]`    | Continuous |
+--------------------+-----------------------+------------+

Hyperparameters for GLMNet
^^^^^^^^^^^^^^^^^^^^^^^^^^

GLMNet is a regularized regression model that supports both LASSO and ridge regression. It is particularly useful for high-dimensional datasets where regularization is necessary to prevent overfitting. The hyperparameters control the strength of the regularization and the balance between L1 and L2 penalties.

+--------------------+---------------------------+-------------+
| **Hyperparameter** |         **Range**         |  **Type**   |
+====================+===========================+=============+
| lambda             | :math:`[0, 10^5]`         | Log-integer |
+--------------------+---------------------------+-------------+
| alpha              | :math:`[0, 1]`            | Continuous  |
+--------------------+---------------------------+-------------+
| nlambda            | :math:`[1, 100]`          | Integer     |
+--------------------+---------------------------+-------------+

Hyperparameters for AlexNet
^^^^^^^^^^^^^^^^^^^^^^^^^^^

AlexNet is a convolutional neural network (CNN) architecture that revolutionized the field of computer vision by achieving significant improvements on the ImageNet dataset. The hyperparameters include learning rate, dropout rate, weight decay, and the choice of activation function, all of which are crucial for training deep neural networks.

+---------------------+----------------------------+-------------+
| **Hyperparameter**  |         **Range**          |  **Type**   |
+=====================+============================+=============+
| learning_rate       | :math:`[10^{-5}, 10^{-1}]` | Continuous  |
+---------------------+----------------------------+-------------+
| dropout_rate        | :math:`[0.0, 0.5]`         | Continuous  |
+---------------------+----------------------------+-------------+
| weight_decay        | :math:`[10^{-5}, 10^{-2}]` | Continuous  |
+---------------------+----------------------------+-------------+
| activation_function | {ReLU, Leaky ReLU, ELU}    | Categorical |
+---------------------+----------------------------+-------------+

Hyperparameters for 2-Layer Bayesian Neural Network (BNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bayesian Neural Networks (BNNs) provide a probabilistic interpretation of deep learning models by introducing uncertainty in the weights. This allows BNNs to express model uncertainty, which is crucial for tasks where uncertainty quantification is important. The hyperparameters include layer sizes, step length, burn-in period, and momentum decay.

+--------------------+----------------------------+----------------+
| **Hyperparameter** |         **Range**          |    **Type**    |
+====================+============================+================+
| layer 1            | :math:`[2^4, 2^9]`         | Log-integer    |
+--------------------+----------------------------+----------------+
| layer 2            | :math:`[2^4, 2^9]`         | Log-integer    |
+--------------------+----------------------------+----------------+
| step_length        | :math:`[10^{-6}, 10^{-1}]` | Log-continuous |
+--------------------+----------------------------+----------------+
| burn_in            | :math:`[0, 8]`             | Integer        |
+--------------------+----------------------------+----------------+
| momentum_decay     | :math:`[0, 1]`             | Log-continuous |
+--------------------+----------------------------+----------------+

Hyperparameters for CNNs
^^^^^^^^^^^^^^^^^^^^^^^^

Convolutional Neural Networks (CNNs) are the backbone of most modern computer vision systems. They are designed to automatically and adaptively learn spatial hierarchies of features through backpropagation. The hyperparameters include learning rate, momentum, regularization parameter, dropout rate, and activation function.

+--------------------------+----------------------------+-------------+
|    **Hyperparameter**    |         **Range**          |  **Type**   |
+==========================+============================+=============+
| learning_rate            | :math:`[10^{-6}, 10^{-1}]` | Continuous  |
+--------------------------+----------------------------+-------------+
| momentum                 | :math:`[0.0, 0.9]`         | Continuous  |
+--------------------------+----------------------------+-------------+
| regularization_parameter | :math:`[10^{-6}, 10^{-2}]` | Continuous  |
+--------------------------+----------------------------+-------------+
| dropout_rate             | :math:`[0, 0.5]`           | Continuous  |
+--------------------------+----------------------------+-------------+
| activation_function      | {ReLU, Leaky ReLU, Tanh,   | Categorical |
|                          | Sigmoid}                   |             |
+--------------------------+----------------------------+-------------+

Hyperparameters for ResNet18
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ResNet18 is a residual network architecture that introduced the concept of residual connections, allowing for the training of very deep networks by mitigating the vanishing gradient problem. The hyperparameters include learning rate, momentum, dropout rate, and weight decay.

+--------------------+----------------------------+------------+
| **Hyperparameter** |         **Range**          |  **Type**  |
+====================+============================+============+
| learning_rate      | :math:`[10^{-5}, 10^{-1}]` | Continuous |
+--------------------+----------------------------+------------+
| momentum           | :math:`[0, 1]`             | Continuous |
+--------------------+----------------------------+------------+
| dropout_rate       | :math:`[0, 0.5]`           | Continuous |
+--------------------+----------------------------+------------+
| weight_decay       | :math:`[10^{-5}, 10^{-2}]` | Continuous |
+--------------------+----------------------------+------------+

Hyperparameters for DenseNet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DenseNet is a densely connected convolutional network that connects each layer to every other layer in a feed-forward fashion. This architecture improves the flow of information and gradients throughout the network, making it easier to train. The hyperparameters include learning rate, momentum, dropout rate, and weight decay.

+--------------------+----------------------------+------------+
| **Hyperparameter** |         **Range**          |  **Type**  |
+====================+============================+============+
| learning_rate      | :math:`[2^3, 2^8]`         | Integer    |
+--------------------+----------------------------+------------+
| momentum           | :math:`[0, 1]`             | Continuous |
+--------------------+----------------------------+------------+
| dropout_rate       | :math:`[0, 0.5]`           | Continuous |
+--------------------+----------------------------+------------+
| weight_decay       | :math:`[10^{-5}, 10^{-1}]` | Continuous |
+--------------------+----------------------------+------------+

Machine Learning Tasks
^^^^^^^^^^^^^^^^^^^^^^

This section lists the various sources of machine learning tasks used for hyperparameter optimization, including classification and regression problems. These datasets are widely recognized in the machine learning community and are used for benchmarking algorithms.

+--------------------------------------------------------+---------------------------+------------+---------+
|                       **Source**                       |         **Type**          | **Number** | **IDs** |
+========================================================+===========================+============+=========+
| `OpenML-CC18 <https://www.openml.org/s/99>`_           | Classification            | 78         | 1-78    |
+--------------------------------------------------------+---------------------------+------------+---------+
| `UC Irvine Repository <https://archive.ics.uci.edu/>`_ | Classification/Regression | 10         | 79-88   |
+--------------------------------------------------------+---------------------------+------------+---------+
| `NAS-Bench-360 <https://archive.ics.uci.edu/>`_        | Classification/Regression | 5          | 89-93   |
+--------------------------------------------------------+---------------------------+------------+---------+
| `NATS-Bench <https://github.com/D-X-Y/NATS-Bench>`_    | Classification            | 3          | 94-96   |
+--------------------------------------------------------+---------------------------+------------+---------+
| `SVHN <https://github.com/D-X-Y/NATS-Bench>`_          | Classification            | 1          | 97      |
+--------------------------------------------------------+---------------------------+------------+---------+


.. _cso-problems:

Configurable Software Optimization Problem
------------------------------------------

This section provides a summary of the configurable software optimization (CSO) tasks, which involve optimizing various software systems. The tasks are characterized by the number of variables, objectives, and workloads, along with the sources of these workloads.

+-------------------+---------------+----------------+---------------+------------------------------------------------------------------------------------------------------------------------------------------+
| **Software Name** | **Variables** | **Objectives** | **Workloads** |                                                           **Workloads Source**                                                           |
+===================+===============+================+===============+==========================================================================================================================================+
| LLVM              | 93            | 8              | 50            | `PolyBench <https://web.cs.ucla.edu/~pouchet/software/polybench/>`_, `mibench <https://github.com/embecosm/mibench?tab=readme-ov-file>`_ |
+-------------------+---------------+----------------+---------------+------------------------------------------------------------------------------------------------------------------------------------------+
| GCC               | 105           | 8              | 50            | `PolyBench <https://web.cs.ucla.edu/~pouchet/software/polybench/>`_, `mibench <https://github.com/embecosm/mibench?tab=readme-ov-file>`_ |
+-------------------+---------------+----------------+---------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Mysql             | 28            | 14             | 18            | `benchbase <https://github.com/cmu-db/benchbase.git>`_, `sysbench <https://github.com/akopytov/sysbench>`_                               |
+-------------------+---------------+----------------+---------------+------------------------------------------------------------------------------------------------------------------------------------------+
| Hadoop            | 206           | 1              | 29            | `HiBench <https://github.com/Intel-bigdata/HiBench>`_                                                                                    |
+-------------------+---------------+----------------+---------------+------------------------------------------------------------------------------------------------------------------------------------------+

.. _rna-problems:

RNA Inverse Design Problem
---------------------------

RNA inverse design involves designing RNA sequences that fold into specific secondary structures. This task is crucial for understanding and manipulating RNA function in various biological processes. The datasets listed here are commonly used benchmarks for RNA design algorithms.

+---------------------------------------------------------------------+-------------------------+-------------+
|                             **Source**                              | **Min-Max Length (nt)** | **Samples** |
+=====================================================================+=========================+=============+
| `Eterna100 <https://github.com/eternagame/eterna100-benchmarking>`_ | 11-399                  | 100         |
+---------------------------------------------------------------------+-------------------------+-------------+
| `Rfam-learn test <https://rfam.org/>`_                              | 50-446                  | 100         |
+---------------------------------------------------------------------+-------------------------+-------------+
| `RNA-Strand <http://www.rnasoft.ca/strand/>`_                       | 4-4381                  | 50          |
+---------------------------------------------------------------------+-------------------------+-------------+
| `RNAStralign <https://github.com/D-X-Y/NATS-Bench>`_                | 30-1851                 | 37149       |
+---------------------------------------------------------------------+-------------------------+-------------+
| `ArchiveII <https://github.com/D-X-Y/NATS-Bench>`_                  | 28-2968                 | 2975        |
+---------------------------------------------------------------------+-------------------------+-------------+


.. _pif-problems:

Protein Inverse Folding Problem
--------------------------------

Protein Inverse Folding involves creating new amino acids sequence folding into desiered backbone structure. These problems are essential for applications in drug design, biotechnology, and synthetic biology. The datasets listed here are widely used in protein inverse folding research.

+--------------------------------------------------------+-----------------------------+-------------+
|                       **Source**                       |          **Type**           | **Numbers** |
+========================================================+=============================+=============+
| `Absolute <https://github.com/csi-greifflab/Absolut>`_ | Antibody design             | 159         |
+--------------------------------------------------------+-----------------------------+-------------+
| `CATH <https://www.cathdb.info/>`_                     | Single-chain protein design | 19752       |
+--------------------------------------------------------+-----------------------------+-------------+
| `Protein Data Bank <https://www.rcsb.org/>`_           | Multi-chain protein design  | 26361       |
+--------------------------------------------------------+-----------------------------+-------------+

.. _parallelization:

Parallelization
---------------

To-do
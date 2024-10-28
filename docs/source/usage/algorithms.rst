Algorithmic objects
===================

.. admonition:: Overview
   :class: info
   
   - :ref:`Register <register-new-algorithm>`: How to register a new algorithmic Object to :ref:`TransOPT <home>`
   - :ref:`Supported Algorithms <alg>`: The list of the synthetic problems available in :ref:`TransOPT <home>`
   - :ref:`Algorithmic Objects<alg-obj>`: The list of the protein inverse folding problems available in :ref:`TransOPT <home>`


.. _register-new-algorithm:

Registering a New Algorithm in TransOPT
---------------------------------------

To register a new algorithm object in TransOPT, follow the steps outlined below:

1. **Import the Model Registry**

   First, you need to import the `model_registry` from the `transopt.agent.registry` module:

   .. code-block:: python

      from transopt.agent.registry import model_registry

2. **Define the Algorithm Object Name**

   Next, use the registry to define the name of your algorithm object. For example:

   .. code-block:: python

      @model_registry.register("MHGP")
      class MHGP(Model):
          pass

   In this example, the algorithm object is named "MHGP".

3. **Choose the Appropriate Base Class**

   Depending on the type of algorithm object you are creating, you must inherit from a specific base class. TransOPT provides several algorithm modules, each corresponding to a different base class:

   - **Surrogate Model**: Inherit from the `Model` class.
   - **Initialization Design**: Inherit from the `Sampler` class.
   - **Acquisition Function**: Inherit from the `AcquisitionBase` class.
   - **Pretrain Module**: Inherit from the `PretrainBase` class.
   - **Normalizer Module**: Inherit from the `NormalizerBase` class.

   For instance, in the example provided, we are creating a surrogate model, so the `MHGP` class inherits from the `Model` base class.

4. **Implement the Required Abstract Methods**

   Once the class is defined, you need to implement several abstract methods that are required by the `Model` base class. These methods include:

   .. code-block:: python

      def meta_fit(
          self,
          source_X : List[np.ndarray],
          source_Y : List[np.ndarray],
          optimize: Union[bool, Sequence[bool]] = True,
      ):
          pass

      def fit(
          self,
          X: np.ndarray,
          Y: np.ndarray,
          optimize: bool = False,
      ):
          pass

      def predict(
          self, X: np.ndarray, return_full: bool = False, with_noise: bool = False
      ) -> Tuple[np.ndarray, np.ndarray]:
          pass

   - **meta_fit**: This method is used to fit meta-data. If your transfer optimization algorithm requires meta-data, this is where you should leverage it.
   - **fit**: This method is used to fit the data for the current task.

By following these steps, you can successfully register a new algorithm object in TransOPT and implement the necessary functionality to integrate it into the framework.



.. _alg:

Supported Algorithms
--------------------

Search space transform
^^^^^^^^^^^^^^^^^^^^^^
**Hyperparameter Search Space Pruning – A New Component for Sequential Model-Based Hyperparameter Optimization**:cite:`WistubaSS15b`

This method prunes ineffective regions of the hyperparameter search space by using past evaluations to guide the optimization. It identifies areas with low potential by analyzing the performance of sampled configurations and employing a surrogate model to predict future outcomes. Regions that consistently show poor performance or low expected improvement are marked as low potential. The method then updates the search process to focus on more promising regions, thereby improving optimization efficiency and reducing unnecessary evaluations.

**Learning search spaces for Bayesian optimization- Another view of hyperparameter transfer learning**:cite:`PerroneS19`

The method replaces predefined search space with data-driven geometrical representations (e.g., ellipsoids and boxes) by analyzing historical data to identify high-performing regions and fitting these regions with geometrical shapes. This transformation narrows the search to promising areas, improving efficiency as the search space dimension increases.

Initialization Design
^^^^^^^^^^^^^^^^^^^^^^
**FEW-SHOT BAYESIAN OPTIMIZATION WITH DEEP KERNEL SURROGATES**:cite:`WistubaG21`

This method leverages historical task data and an evolutionary algorithm to provide a warm-start initialization. By selecting hyperparameter settings that minimize a loss function across multiple tasks, the method accelerates optimization with fewer evaluations. 

**Initializing Bayesian Hyperparameter Optimization via Meta-Learning**:cite:`FeurerSH15`

This method introduces a meta-learning-based initialization for BO, improving the starting point by leveraging hyperparameter configurations that worked well on similar datasets. These similar datasets are identified through meta-features. The method calculates the distance between datasets using these meta-features, selecting the most similar ones to initialize the optimization process efficiently.

**Learning Hyperparameter Optimization Initializations**:cite:`WistubaSS15a`

This method proposes to use a meta-loss function that is minimized through gradient-based optimization. By optimizing for a meta-loss derived from the response functions of past datasets, it generates entirely new configurations, whereas prior methods limited themselves to reusing configurations in similar datasets.

Surrogate Model
^^^^^^^^^^^^^^^^^^^^^^
**Pre-trained Gaussian processes for Bayesian optimization**:cite:`Wang2021`

In this method, the surrogate model is built on a pre-trained GP with data from related tasks. This approach uses a KL divergence-based loss function to pre-train the GP, ensuring it captures similarities between the target function and past data. The pre-trained GP serves as the prior for BO, allowing the model to make better predictions with fewer observations by leveraging the pre-trained knowledge.

**FEW-SHOT BAYESIAN OPTIMIZATION WITH DEEP KERNEL SURROGATES**

In this method, the surrogate model is a deep kernel Gaussian process that is meta-learned across multiple past tasks. This model enables quick adaptation to new tasks with limited evaluations. The deep kernel, which combines a neural network and a Gaussian process, provides uncertainty estimates, helping the model generalize across diverse tasks while being fine-tuned for new ones.

**Google Vizier- A Service for Black-Box Optimization**:cite:`GolovinSMKKS17`

This method transfers source knowledge by using the posterior mean of the source task as the prior mean for the target task. This approach simplifies the transfer process by ignoring uncertainty from the source model and only leveraging the mean, which leads to reduced computational complexity while still incorporating valuable information from the source task. 

**PFNs4BO- In-Context Learning for Bayesian Optimization**:cite:`MullerFHH23`

This method utilizes a Transformer-based architecture called Prior-data Fitted Networks (PFNs). These networks are trained on synthetic datasets to approximate the posterior predictive distribution (PPD) through in-context learning. PFNs can be trained on any efficiently sampled prior distribution, such as Gaussian processes or Bayesian neural networks. By learning from diverse priors, the PFN surrogate model captures complex patterns in the optimization process, allowing it to make accurate predictions while maintaining flexibility to incorporate user-defined priors or handle spurious dimensions effectively.

**Scalable Gaussian process-based transfer surrogates for hyperparameter optimization**:cite:`WistubaSS18`

This method introduces an ensemble of GP, where each GP is trained on a different past task. The model uses a weighted sum approach to combine the predictions from each GP. The weights are assigned based on how well each GP predicts the target task, with more relevant models receiving higher weights. 

**Scalable Meta-Learning for Bayesian Optimization using Ranking-Weighted Gaussian Process Ensembles**:cite:`FeurerBE15`

This method introduces Ranking-Weighted Gaussian Process Ensembles (RGPE). Similar to previous approaches, the surrogate model combines an ensemble of GPs. However, in RGPE, the weights are determined using a ranking loss function, which assesses how effectively each GP ranks the observations from the current task. GPs that rank the observations more accurately are assigned higher weights, reflecting their greater relevance to the task at hand.

**Multi-Task Bayesian Optimization**:cite:`SwerskySA13`

This method uses multi-task Gaussian processes (MTGP) as the surrogate model. It trains a GP for each task and uses a shared covariance structure across tasks to improve predictive accuracy. By leveraging the relationships between tasks, the MTGP reduces the need for independent function evaluations, making the optimization process faster and more efficient.

**Multi-Fidelity Bayesian Optimization via Deep Neural Networks**:cite:`LiXKZ20`

In this method, the surrogate model employs a deep neural network designed to handle multi-fidelity optimization tasks. The DNN surrogate models each fidelity with a neural network, and higher fidelities are conditioned on the outputs from lower fidelities. By stacking neural networks for each fidelity level, the model captures nonlinear relationships between different fidelities. This structure allows the surrogate to propagate information across fidelities, improving the accuracy of function estimation at higher fidelities while reducing computational costs.

**BOHB: robust and efficient hyperparameter optimization at scale**:cite:`FalknerKH18`

In this method, the surrogate model uses a Tree-structured Parzen Estimator (TPE) to model the hyperparameter space. TPE builds separate probability models for good and bad configurations using kernel density estimation. The TPE model guides the search by maximizing the ratio between these models, effectively focusing on promising regions of the search space. 

Acquisition Function
^^^^^^^^^^^^^^^^^^^^
**Scalable Meta-Learning for Bayesian Optimization using Ranking-Weighted Gaussian Process Ensembles**

In RGPE, the acquisition function follows standard BO methods but integrates the ranking-weighted ensemble model. The ensemble combines predictions from multiple GPs, each weighted based on its ranking performance in relation to the current task. The acquisition function then uses this weighted ensemble to balance exploration and exploitation, ensuring that the most relevant past models are given greater influence when selecting the next point to evaluate 

**Scalable Gaussian process-based transfer surrogates for hyperparameter optimization**

This approach is referred to as the *transfer acquisition function* (TAF). The acquisition function balances exploration and exploitation by combining the predicted improvement from the new data with predicted improvements from previous tasks, weighted by their relevance. The weights are calculated the same as the model.

**Multi-Task Bayesian Optimization**

In this method, the acquisition function extends the standard EI criterion to the multi-task setting. It dynamically selects which task to evaluate by considering the correlation between tasks. The acquisition function maximizes information gain per unit cost by balancing the evaluation of cheaper auxiliary tasks with more expensive primary tasks, using the entropy search strategy. 

**Multi-Fidelity Bayesian Optimization via Deep Neural Networks**

It aims to maximize the mutual information between the predicted maximum of the objective function and the next point to be evaluated. The acquisition function selects the input location and fidelity level that provide the highest benefit-cost ratio. By employing fidelity-wise moment matching and Gauss-Hermite quadrature to approximate the posterior distributions, the acquisition function ensures that both fidelity selection and input sampling are computationally efficient and well-informed.

**BOHB:Robust and Efficient Hyperparameter Optimization at Scale**

It selects new configurations by maximizing the expected improvement, using kernel density estimates of good and bad configurations. BOHB combines this with a multi-fidelity approach, which allows the acquisition function to operate across different budget levels, efficiently balancing exploration and exploitation while scaling to large optimization tasks

**Reinforced Few-Shot Acquisition Function Learning for Bayesian Optimization**:cite:`HsiehHL21`

In this method, the acquisition function is modeled with a deep Q-network (DQN), learning to balance exploration and exploitation as a reinforcement learning task. The DQN predicts sampling utility based on the posterior mean and variance, refined by a Bayesian variant that incorporates uncertainty to avoid overfitting.




.. _alg-obj:

List of Algorithmic Objects
---------------------------
The optimization framework includes a variety of state-of-the-art algorithms, each designed with specific features to address different classes of optimization problems. The table below provides a summary of the key algorithms available, categorized by their class, convenience for use, targeted objective(s), and any constraints they impose.

.. csv-table::
   :header: "Algorithmic Objects", "Type", "Source Algorithm"
   :widths: 60, 10, 100
   :file: algorithms.csv


References
----------

.. bibliography:: TOS.bib
   :style: plain
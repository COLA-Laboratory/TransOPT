Visualization
===============

This section demonstrates various visualization techniques used in TransOPT.

Data Filtering and Statistical Visualization
--------------------------------------------

This section demonstrates how to filter data into multiple groups based on different conditions, perform statistical analysis on these groups, and visualize the results using box plots and trajectory plots.

.. figure:: /_static/figures/visualization/filter.jpeg
   :alt: Data Filtering Process
   :width: 100%
   :align: center

   Figure 1: Four groups with different surrogate model. 

The above figure illustrates the process of filtering data into multiple groups based on different conditions. This visual representation helps to understand how the data is segmented and analyzed in our visualization approach.

Key steps in the data filtering process:

1. Click + to add a new filter group.
2. Define filter conditions for each group.
3. Apply filters and generate visualizations (e.g., box plots, trajectory plots) for each group


Visualization of Filtered Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After filtering the data into groups, TransOPT provides two main types of visualizations to compare and analyze the results: trajectory plots and box plots.

Trajectory Plot
"""""""""""""""

The trajectory plot shows the performance of different groups over time or iterations.

.. figure:: /_static/figures/visualization/traj_compare.jpg
   :alt: Trajectory Plot of Different Groups
   :width: 50%
   :align: center

   Figure 2: Trajectory plot comparing performance of different surrogate model groups over iterations.

This plot allows you to:

- Compare the convergence rates of different groups
- Identify which group performs better at different stages of the optimization process
- Observe any significant differences in performance trends among the groups

Box Plot
""""""""

The box plot provides a statistical summary of the performance distribution for each group.

.. figure:: /_static/figures/visualization/box_compare.jpg
   :alt: Box Plot of Different Groups
   :width: 50%
   :align: center

   Figure 3: Box plot showing performance distribution of different surrogate model groups.

Key insights from the box plot:

- Median performance of each group
- Spread of performance within each group
- Presence of any outliers
- Easy comparison of performance distributions across groups




Analysis of Individual Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TransOPT also provides tools for in-depth analysis of individual datasets. This section outlines the process and visualizations available for single dataset analysis.

1. Dataset Selection
""""""""""""""""""""

The first step is to select a specific dataset for analysis. Once selected, TransOPT generates a summary of the dataset's key information.

.. figure:: /_static/figures/visualization/choose.jpg
   :alt: Dataset Information Summary
   :width: 100%
   :align: center

   Figure 4: Summary of selected dataset information, including algorithm modules used and optimization problem details.


2. Trajectory Plot
""""""""""""""""""

The trajectory plot for the selected dataset shows the optimization performance over time or iterations.

.. figure:: /_static/figures/visualization/traj_solo.jpg
   :alt: Trajectory Plot of Single Dataset
   :width: 50%
   :align: center

   Figure 5: Trajectory plot showing the optimization performance for the selected dataset.

This visualization allows users to:

- Observe the convergence behavior of the optimization process
- Identify any plateaus or sudden improvements in performance
- Assess the overall efficiency of the optimization algorithm for this specific dataset

3. Variable Importance
""""""""""""""""""""""

The variable importance plot highlights which features or parameters had the most significant impact on the optimization outcome.

.. figure:: /_static/figures/visualization/importance.jpg
   :alt: Variable Importance Plot
   :width: 50%
   :align: center

   Figure 6: Variable importance plot showing the relative impact of different features or parameters.

This visualization helps users:

- Identify the most influential variables in the optimization process
- Understand which parameters might require more careful tuning
- Gain insights into the underlying structure of the optimization problem

4. Dimensionality Reduction Plot
""""""""""""""""""""""""""""""""

The dimensionality reduction plot provides a 2D representation of the high-dimensional sampling data, typically using techniques like PCA or t-SNE.

.. figure:: /_static/figures/visualization/footprint.jpg
   :alt: Dimensionality Reduction Plot
   :width: 50%
   :align: center

   Figure 7: 2D plot of the sampled data after dimensionality reduction.

This visualization allows users to:

- Observe clusters or patterns in the sampling data
- Identify regions of the search space that were more heavily explored
- Gain intuition about the structure of the optimization landscape



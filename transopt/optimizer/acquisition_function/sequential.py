from GPyOpt.core.evaluators.base import EvaluatorBase


class Sequential(EvaluatorBase):
    """
    Class for standard Sequential Bayesian optimization methods.

    :param acquisition: acquisition function to be used to compute the batch.
    :param batch size: it is 1 by default since this class is only used for sequential methods.
    """

    def __init__(self, acquisition, batch_size=1):
        super(Sequential, self).__init__(acquisition, batch_size)

    def compute_batch(self, duplicate_manager=None,context_manager=None):
        """
        Selects the new location to evaluate the objective.
        """
        x, acq_value = self.acquisition.optimize(duplicate_manager=duplicate_manager)
        return x, acq_value


# class Sequential_Tabular(EvaluatorBase):
#     """
#     Class for standard Sequential Bayesian optimization methods.
#
#     :param acquisition: acquisition function to be used to compute the batch.
#     :param batch size: it is 1 by default since this class is only used for sequential methods.
#     """
#
#     def __init__(self, acquisition, batch_size=1):
#         super(Sequential_Tabular, self).__init__(acquisition, batch_size)
#
#     def compute_batch(self, X, unobserved_indexes):
#         """
#         Selects the new location to evaluate the objective.
#         """
#         acq_value = self.acquisition._compute_acq(X)
#         min_index = np.argmin(acq_value)
#         return unobserved_indexes[min_index], acq_value[min_index]
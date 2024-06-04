import logging
import os
from copy import deepcopy
from typing import Dict, Tuple

import gpytorch
import numpy as np
import torch
import torch.nn as nn
from torch import cat

from transopt.agent.registry import model_registry


class FeatureExtractor(nn.Module):
    """
    The feature extractor that is part of the deep kernel.
    """
    def __init__(self, configuration):
        super(FeatureExtractor, self).__init__()

        self.configuration = configuration

        self.nr_layers = configuration['nr_layers']
        self.act_func = nn.LeakyReLU()
        # adding one to the dimensionality of the initial input features
        # for the concatenation with the budget.
        initial_features = configuration['nr_initial_features'] + 1
        self.fc1 = nn.Linear(initial_features, configuration['layer1_units'])
        self.bn1 = nn.BatchNorm1d(configuration['layer1_units'])
        for i in range(2, self.nr_layers):
            setattr(
                self,
                f'fc{i + 1}',
                nn.Linear(configuration[f'layer{i - 1}_units'], configuration[f'layer{i}_units']),
            )
            setattr(
                self,
                f'bn{i + 1}',
                nn.BatchNorm1d(configuration[f'layer{i}_units']),
            )


        setattr(
            self,
            f'fc{self.nr_layers}',
            nn.Linear(
                configuration[f'layer{self.nr_layers - 1}_units'] +
                configuration['cnn_nr_channels'],  # accounting for the learning curve features
                configuration[f'layer{self.nr_layers}_units']
            ),
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, kernel_size=(configuration['cnn_kernel_size'],), out_channels=4),
            nn.AdaptiveMaxPool1d(1),
        )

    def forward(self, x, budgets, learning_curves):

        # add an extra dimensionality for the budget
        # making it nr_rows x 1.
        budgets = torch.unsqueeze(budgets, dim=1)
        # concatenate budgets with examples
        x = cat((x, budgets), dim=1)
        x = self.fc1(x)
        x = self.act_func(self.bn1(x))

        for i in range(2, self.nr_layers):
            x = self.act_func(
                getattr(self, f'bn{i}')(
                    getattr(self, f'fc{i}')(
                        x
                    )
                )
            )

        # add an extra dimensionality for the learning curve
        # making it nr_rows x 1 x lc_values.
        learning_curves = torch.unsqueeze(learning_curves, 1)
        lc_features = self.cnn(learning_curves)
        # revert the output from the cnn into nr_rows x nr_kernels.
        lc_features = torch.squeeze(lc_features, 2)

        # put learning curve features into the last layer along with the higher level features.
        x = cat((x, lc_features), dim=1)
        x = self.act_func(getattr(self, f'fc{self.nr_layers}')(x))

        return x


class GPRegressionModel(gpytorch.models.ExactGP):
    """
    A simple GP model.
    """
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        """
        Constructor of the GPRegressionModel.

        Args:
            train_x: The initial train examples for the GP.
            train_y: The initial train labels for the GP.
            likelihood: The likelihood to be used.
        """
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DyHPO:
    """
    The DyHPO DeepGP model.
    """
    def __init__(
        self,
        configuration: Dict,
        device: torch.device,
        dataset_name: str = 'unknown',
        output_path: str = '.',
        seed: int = 11,
    ):
        """
        The constructor for the DyHPO model.

        Args:
            configuration: The configuration to be used
                for the different parts of the surrogate.
            device: The device where the experiments will be run on.
            dataset_name: The name of the dataset for the current run.
            output_path: The path where the intermediate/final results
                will be stored.
            seed: The seed that will be used to store the checkpoint
                properly.
        """
        super(DyHPO, self).__init__()
        self.feature_extractor = FeatureExtractor(configuration)
        self.batch_size = configuration['batch_size']
        self.nr_epochs = configuration['nr_epochs']
        self.early_stopping_patience = configuration['nr_patience_epochs']
        self.refine_epochs = 50
        self.dev = device
        self.seed = seed
        self.model, self.likelihood, self.mll = \
            self.get_model_likelihood_mll(
                configuration[f'layer{self.feature_extractor.nr_layers}_units']
            )

        self.model.to(self.dev)
        self.likelihood.to(self.dev)
        self.feature_extractor.to(self.dev)

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': configuration['learning_rate']},
            {'params': self.feature_extractor.parameters(), 'lr': configuration['learning_rate']}],
        )

        self.configuration = configuration
        # the number of initial points for which we will retrain fully from scratch
        # This is basically equal to the dimensionality of the search space + 1.
        self.initial_nr_points = 10
        # keeping track of the total hpo iterations. It will be used during the optimization
        # process to switch from fully training the model, to refining.
        self.iterations = 0
        # flag for when the optimization of the model should start from scratch.
        self.restart = True

        self.logger = logging.getLogger(__name__)

        self.checkpoint_path = os.path.join(
            output_path,
            'checkpoints',
            f'{dataset_name}',
            f'{self.seed}',
        )

        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.checkpoint_file = os.path.join(
            self.checkpoint_path,
            'checkpoint.pth'
        )

    def restart_optimization(self):
        """
        Restart the surrogate model from scratch.
        """
        self.feature_extractor = FeatureExtractor(self.configuration).to(self.dev)
        self.model, self.likelihood, self.mll = \
            self.get_model_likelihood_mll(
                self.configuration[f'layer{self.feature_extractor.nr_layers}_units'],
            )

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.configuration['learning_rate']},
            {'params': self.feature_extractor.parameters(), 'lr': self.configuration['learning_rate']}],
        )

    def get_model_likelihood_mll(
        self,
        train_size: int,
    ) -> Tuple[GPRegressionModel, gpytorch.likelihoods.GaussianLikelihood, gpytorch.mlls.ExactMarginalLogLikelihood]:
        """
        Called when the surrogate is first initialized or restarted.

        Args:
            train_size: The size of the current training set.

        Returns:
            model, likelihood, mll - The GP model, the likelihood and
                the marginal likelihood.
        """
        train_x = torch.ones(train_size, train_size).to(self.dev)
        train_y = torch.ones(train_size).to(self.dev)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.dev)
        model = GPRegressionModel(train_x=train_x, train_y=train_y, likelihood=likelihood).to(self.dev)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.dev)

        return model, likelihood, mll

    def train_pipeline(self, data: Dict[str, torch.Tensor], load_checkpoint: bool = False):
        """
        Train the surrogate model.

        Args:
            data: A dictionary which has the training examples, training features,
                training budgets and in the end the training curves.
            load_checkpoint: A flag whether to load the state from a previous checkpoint,
                or whether to start from scratch.
        """
        self.iterations += 1
        self.logger.debug(f'Starting iteration: {self.iterations}')
        # whether the state has been changed. Basically, if a better loss was found during
        # this optimization iteration then the state (weights) were changed.
        weights_changed = False

        if load_checkpoint:
            try:
                self.load_checkpoint()
            except FileNotFoundError:
                self.logger.error(f'No checkpoint file found at: {self.checkpoint_file}'
                                  f'Training the GP from the beginning')

        self.model.train()
        self.likelihood.train()
        self.feature_extractor.train()

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.configuration['learning_rate']},
            {'params': self.feature_extractor.parameters(), 'lr': self.configuration['learning_rate']}],
        )

        X_train = data['X_train']
        train_budgets = data['train_budgets']
        train_curves = data['train_curves']
        y_train = data['y_train']

        initial_state = self.get_state()
        training_errored = False

        if self.restart:
            self.restart_optimization()
            nr_epochs = self.nr_epochs
            # 2 cases where the statement below is hit.
            # - We are switching from the full training phase in the beginning to refining.
            # - We are restarting because our refining diverged
            if self.initial_nr_points <= self.iterations:
                self.restart = False
        else:
            nr_epochs = self.refine_epochs

        # where the mean squared error will be stored
        # when predicting on the train set
        mse = 0.0

        for epoch_nr in range(0, nr_epochs):

            nr_examples_batch = X_train.size(dim=0)
            # if only one example in the batch, skip the batch.
            # Otherwise, the code will fail because of batchnorm
            if nr_examples_batch == 1:
                continue

            # Zero backprop gradients
            self.optimizer.zero_grad()

            projected_x = self.feature_extractor(X_train, train_budgets, train_curves)
            self.model.set_train_data(projected_x, y_train, strict=False)
            output = self.model(projected_x)

            try:
                # Calc loss and backprop derivatives
                loss = -self.mll(output, self.model.train_targets)
                loss_value = loss.detach().to('cpu').item()
                mse = gpytorch.metrics.mean_squared_error(output, self.model.train_targets)
                self.logger.debug(
                    f'Epoch {epoch_nr} - MSE {mse:.5f}, '
                    f'Loss: {loss_value:.3f}, '
                    f'lengthscale: {self.model.covar_module.base_kernel.lengthscale.item():.3f}, '
                    f'noise: {self.model.likelihood.noise.item():.3f}, '
                )
                loss.backward()
                self.optimizer.step()
            except Exception as training_error:
                self.logger.error(f'The following error happened while training: {training_error}')
                # An error has happened, trigger the restart of the optimization and restart
                # the model with default hyperparameters.
                self.restart = True
                training_errored = True
                break

        """
        # metric too high, time to restart, or we risk divergence
        if mse > 0.15:
            if not self.restart:
                self.restart = True
        """
        if training_errored:
            self.save_checkpoint(initial_state)
            self.load_checkpoint()

    def predict_pipeline(
        self,
        train_data: Dict[str, torch.Tensor],
        test_data: Dict[str, torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        Args:
            train_data: A dictionary that has the training
                examples, features, budgets and learning curves.
            test_data: Same as for the training data, but it is
                for the testing part and it does not feature labels.

        Returns:
            means, stds: The means of the predictions for the
                testing points and the standard deviations.
        """
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad(): # gpytorch.settings.fast_pred_var():
            projected_train_x = self.feature_extractor(
                train_data['X_train'],
                train_data['train_budgets'],
                train_data['train_curves'],
            )
            self.model.set_train_data(inputs=projected_train_x, targets=train_data['y_train'], strict=False)
            projected_test_x = self.feature_extractor(
                test_data['X_test'],
                test_data['test_budgets'],
                test_data['test_curves'],
            )
            preds = self.likelihood(self.model(projected_test_x))

        means = preds.mean.detach().to('cpu').numpy().reshape(-1, )
        stds = preds.stddev.detach().to('cpu').numpy().reshape(-1, )

        return means, stds

    def load_checkpoint(self):
        """
        Load the state from a previous checkpoint.
        """
        checkpoint = torch.load(self.checkpoint_file)
        self.model.load_state_dict(checkpoint['gp_state_dict'])
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

    def save_checkpoint(self, state: Dict =None):
        """
        Save the given state or the current state in a
        checkpoint file.

        Args:
            state: The state to save, if none, it will
            save the current state.
        """

        if state is None:
            torch.save(
                self.get_state(),
                self.checkpoint_file,
            )
        else:
            torch.save(
                state,
                self.checkpoint_file,
            )

    def get_state(self) -> Dict[str, Dict]:
        """
        Get the current state of the surrogate.

        Returns:
            current_state: A dictionary that represents
                the current state of the surrogate model.
        """
        current_state = {
            'gp_state_dict': deepcopy(self.model.state_dict()),
            'feature_extractor_state_dict': deepcopy(self.feature_extractor.state_dict()),
            'likelihood_state_dict': deepcopy(self.likelihood.state_dict()),
        }

        return current_state
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np


def get_hparams(algorithm, dataset, random_seed, model_size=None, architecture='resnet'):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    hparams = {}
    hparam_space = get_hparam_space(algorithm, model_size, architecture)
    random_state = np.random.RandomState(random_seed)

    for name, (hparam_type, range_or_values) in hparam_space.items():
        if hparam_type == 'categorical':
            default_val = range_or_values[0]
            random_val = random_state.choice(range_or_values)
        elif hparam_type == 'float':
            default_val = sum(range_or_values) / 2
            random_val = random_state.uniform(*range_or_values)
        elif hparam_type == 'int':
            default_val = int(sum(range_or_values) / 2)
            random_val = random_state.randint(*range_or_values)
        elif hparam_type == 'log':
            default_val = 10 ** (sum(range_or_values) / 2)
            random_val = 10 ** random_state.uniform(*range_or_values)
        else:
            raise ValueError(f"Unknown hparam type: {hparam_type}")

        hparams[name] = (default_val, random_val)

    return hparams

def default_hparams(algorithm, dataset, model_size='small', architecture='resnet'):
    return {a: b for a, (b, c) in get_hparams(algorithm, dataset, 0, model_size, architecture).items()}

def random_hparams(algorithm, dataset, seed, model_size='small', architecture='resnet'):
    return {a: c for a, (b, c) in get_hparams(algorithm, dataset, seed, model_size, architecture).items()}

def get_hparam_space(algorithm, model_size=None, architecture='resnet'):
    """
    Returns a dictionary of hyperparameter spaces for the given algorithm and dataset.
    Each entry is a tuple of (type, range) where type is 'float', 'int', or 'categorical'.
    """
    hparam_space = {}

    if algorithm in ['ERM', 'GLMNet', 'BayesianNN']:
        hparam_space['lr'] = ('log', (-6, -2))
        hparam_space['weight_decay'] = ('log', (-7, -4))
        hparam_space['momentum'] = ('float', (0.5, 0.999))
        hparam_space['batch_size'] = ('categorical', [16, 32, 64, 128])

    if algorithm == 'ERM':
        # hparam_space['batch_size'] = ('categorical', [16, 32, 64, 128])
        hparam_space['dropout_rate'] = ('float', (0, 0.5))
        if architecture.lower() == 'cnn':
            hparam_space['hidden_dim1'] = ('categorical', [32, 64, 128])
            hparam_space['hidden_dim2'] = ('categorical', [32, 64, 128])

    if algorithm == 'GLMNet':
        hparam_space['glmnet_alpha'] = ('log', (-4, 1))
        hparam_space['glmnet_l1_ratio'] = ('float', (0, 1))

    if algorithm == 'BayesianNN':
        hparam_space['bayesian_num_samples'] = ('categorical', [5, 10, 20, 50])
        hparam_space['bayesian_hidden_dim1'] = ('categorical', [32, 64, 128, 256])
        hparam_space['bayesian_hidden_dim2'] = ('categorical', [32, 64, 128, 256])
        hparam_space['step_length'] = ('log', (-4, -1))
        hparam_space['burn_in'] = ('categorical', [500, 1000, 2000, 5000])

    # Add hidden dimensions for CNN architecture



    return hparam_space

def test_hparam_registry():
    algorithms = ['ERM', 'GLMNet', 'BayesianNN']
    datasets = ['RobCifar10', 'RobCifar100', 'RobImageNet']
    architectures = ['resnet', 'wideresnet', 'densenet', 'alexnet', 'cnn']

    for algorithm in algorithms:
        for dataset in datasets:
            print(f"\nTesting: Algorithm={algorithm}, Dataset={dataset}")

            # Get default hyperparameters
            default_hparam = default_hparams(algorithm, dataset)
            print("\nDefault hyperparameters:")
            for hparam, value in default_hparam.items():
                print(f"  {hparam}: {value}")

            # Get random hyperparameters
            random_hparam = random_hparams(algorithm, dataset, seed=42)
            print("\nRandom hyperparameters:")
            for hparam, value in random_hparam.items():
                print(f"  {hparam}: {value}")

            # Get hyperparameter space
            hparam_space = get_hparam_space(algorithm, dataset)
            print("\nHyperparameter space:")
            for hparam, (htype, hrange) in hparam_space.items():
                print(f"  {hparam}: type={htype}, range={hrange}")

            print("\n" + "="*50)

if __name__ == "__main__":
    test_hparam_registry()
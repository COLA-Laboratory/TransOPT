import copy

import numpy as np
import scipy
import statsmodels.api as sm
import dask


class KDEMultivariate(sm.nonparametric.KDEMultivariate):
    def __init__(self, configurations):
        self.configurations = configurations
        data = []
        for config in configurations:
            data.append(np.array(config.to_list()))
        data = np.array(data)
        super().__init__(data, configurations[0].kde_vartypes, 'normal_reference')


class Log():
    def __init__(self, size):
        self.size = size
        self.logs = np.empty(self.size, dtype=dict)
        self.best = {'loss': np.inf}

    def __getitem__(self, index):
        return self.logs[index]

    def __setitem__(self, index, value):
        self.logs[index] = value

    def __repr__(self):
        string = []
        string.append(f's_max: {self.size}')
        for s, log in enumerate(self.logs):
            string.append(f's: {s}')
            for budget in log:
                string.append(f'Budget: {budget}')
                string.append(f'Loss: {log[budget]["loss"]}')
                string.append(str(log[budget]['hyperparameter']))
        string.append('Best Hyperparameter Configuration:')
        string.append(f'Budget: {self.best["budget"]}')
        string.append(f'Loss: {self.best["loss"]}')
        string.append(str(self.best['hyperparameter']))
        return '\n'.join(string)


class BOHB:
    def __init__(self, configspace, evaluate, max_budget, min_budget,
                 eta=3, best_percent=0.15, random_percent=1/3, n_samples=64,
                 bw_factor=3, min_bandwidth=1e-3, n_proc=1):
        self.eta = eta
        self.configspace = configspace
        self.max_budget = max_budget
        self.min_budget = min_budget
        self.evaluate = evaluate

        self.best_percent = best_percent
        self.random_percent = random_percent
        self.n_samples = n_samples
        self.min_bandwidth = min_bandwidth
        self.bw_factor = bw_factor
        self.n_proc = n_proc

        self.s_max = int(np.log(self.max_budget/self.min_budget) / np.log(self.eta))
        self.budget = (self.s_max + 1) * self.max_budget

        self.kde_good = None
        self.kde_bad = None
        self.samples = np.array([])

    def optimize(self):
        logs = Log(self.s_max+1)
        for s in reversed(range(self.s_max + 1)):
            logs[s] = {}
            n = int(np.ceil(
                (self.budget * (self.eta ** s)) / (self.max_budget * (s + 1))))
            r = self.max_budget * (self.eta ** -s)
            self.kde_good = None
            self.kde_bad = None
            self.samples = np.array([])
            for i in range(s+1):
                n_i = n * self.eta ** (-i)  # Number of configs
                r_i = r * self.eta ** (i)  # Budget
                logs[s][r_i] = {'loss': np.inf}

                samples = []
                losses = []
                for j in range(n):
                    sample = self.get_sample()
                    if self.n_proc > 1:
                        loss = dask.delayed(self.evaluate)(sample.to_dict(), int(r_i))
                    else:
                        loss = self.evaluate(sample.to_dict(), int(r_i))
                    samples.append(sample)
                    losses.append(loss)
                if self.n_proc > 1:
                    losses = dask.compute(
                        *losses, scheduler='processes', num_workers=self.n_proc)
                midx = np.argmin(losses)
                logs[s][r_i]['loss'] = losses[midx]
                logs[s][r_i]['hyperparameter'] = samples[midx]

                if logs[s][r_i]['loss'] < logs.best['loss']:
                    logs.best['loss'] = logs[s][r_i]['loss']
                    logs.best['budget'] = r_i
                    logs.best['hyperparameter'] = logs[s][r_i]['hyperparameter']

                n = int(np.ceil(n_i/self.eta))
                idxs = np.argsort(losses)
                self.samples = np.array(samples)[idxs[:n]]
                n_good = int(np.ceil(self.best_percent * len(samples)))
                if n_good > len(samples[0].kde_vartypes) + 2:
                    good_data = np.array(samples)[idxs[:n_good]]
                    bad_data = np.array(samples)[idxs[n_good:]]
                    self.kde_good = KDEMultivariate(good_data)
                    self.kde_bad = KDEMultivariate(bad_data)
                    self.kde_bad.bw = np.clip(
                        self.kde_bad.bw, self.min_bandwidth, None)
                    self.kde_good.bw = np.clip(
                        self.kde_good.bw, self.min_bandwidth, None)
        return logs

    def get_sample(self):
        if self.kde_good is None or np.random.random() < self.random_percent:
            if len(self.samples):
                idx = np.random.randint(0, len(self.samples))
                sample = self.samples[idx]
                self.samples = np.delete(self.samples, idx)
                return sample
            else:
                return self.configspace.sample_configuration()

        # Sample from the good data
        best_tpe_val = np.inf
        for _ in range(self.n_samples):
            idx = np.random.randint(0, len(self.kde_good.configurations))
            configuration = copy.deepcopy(self.kde_good.configurations[idx])
            for hyperparameter, bw in zip(configuration, self.kde_good.bw):
                if hyperparameter.type == cs.Type.Continuous:
                    value = hyperparameter.value
                    bw = bw * self.bw_factor
                    hyperparameter.value = scipy.stats.truncnorm.rvs(
                        -value/bw, (1-value)/bw, loc=value, scale=bw)
                elif hyperparameter.type == cs.Type.Discrete:
                    if np.random.rand() >= (1-bw):
                        idx = np.random.randint(len(hyperparameter.choices))
                        hyperparameter.value = idx
                else:
                    raise NotImplementedError

            tpe_val = (self.kde_bad.pdf(configuration.to_list()) /
                       self.kde_good.pdf(configuration.to_list()))
            if tpe_val < best_tpe_val:
                best_tpe_val = tpe_val
                best_configuration = configuration

        return best_configuration
import numpy as np
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import itertools

class FootPrint:
    def __init__(self, X, range):
        self.X = X
        self.ranges = range
        self.boundary_points = self.get_random_boundary_points(0)
        self.config_ids = np.arange(0, len(self.X) + len(self.boundary_points)).tolist()
        self.n_configs = len(self.config_ids)
        
        self._distance = None
        self._reduced_data = None
        

    def calculate_distances(self):
        """
        Calculate pairwise distances between configurations.

        Parameters:
        X (np.ndarray): Encoded data matrix.

        Returns:
        np.ndarray: Pairwise distances matrix.
        """
        distances = np.zeros((self.n_configs, self.n_configs))
        configs = np.vstack((self.X, self.boundary_points))
        for i in range(self.n_configs):
            for j in range(i + 1, self.n_configs):
                distances[i, j] = distances[j, i] = np.linalg.norm(configs[i] - configs[j])

        self._distances = distances

    def init_distances(self, config_ids, exclude_configs=False):
        """
        Initialize pairwise distances between configurations.

        Parameters:
        X (np.ndarray): Encoded data matrix.
        config_ids (List[int]): Corresponding config_ids.
        exclude_configs (bool): Whether to exclude the passed X. Default is False.

        Returns:
        np.ndarray: Pairwise distances matrix.
        """
        if not exclude_configs:
            self.calculate_distances()
        else:
            return np.zeros((0, 0))

    def update_distances(self, X, distances, config, rejection_threshold=0.0):
        """
        Update pairwise distances with a new configuration.

        Parameters:
        X (np.ndarray): Encoded data matrix.
        distances (np.ndarray): Pairwise distances matrix.
        config (np.ndarray): New configuration to add.
        rejection_threshold (float): Threshold for rejecting the config. Default is 0.0.

        Returns:
        bool: Whether the config was rejected or not.
        """
        n_configs = X.shape[0]
        new_distances = np.zeros((n_configs + 1, n_configs + 1))
        rejected = False

        if n_configs > 0:
            new_distances[:n_configs, :n_configs] = distances[:, :]
            for j in range(n_configs):
                d = np.linalg.norm(X[j] - config)
                if rejection_threshold is not None:
                    if d < rejection_threshold:
                        rejected = True
                        break

                new_distances[n_configs, j] = new_distances[j, n_configs] = d

        if not rejected:
            X = np.vstack((X, config))
            distances = new_distanceslist

        return rejected
        
    def get_random_boundary_points(self, num_samples):
        num_dims = len(self.ranges)
    
        combinations = list(itertools.product(*self.ranges))
        
        # random_boundary_indices = np.random.choice(len(combinations), num_samples, replace=False)
        # random_boundary_points = [combinations[i] for i in random_boundary_indices]

        return np.array(combinations)

        
    def get_mds(self):

        if self._distances is None:
            raise RuntimeError("You need to call `calculate` first.")

        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
        self._reduced_data =  mds.fit_transform(self._distances).tolist()
    
    def plot_embedding(self):
        """
        Plot the low-dimensional embedding.

        """
        plt.figure(figsize=(8, 6))
        plt.scatter(self._reduced_data[:len(self.X), 0], self._reduced_data[:len(self.X), 1], c='b', label='MDS Embedding')
        plt.scatter(self._reduced_data[len(self.X):, 0], self._reduced_data[len(self.X):, 1], c='r', marker= 'x', label='Boundary  points')

        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title('MDS Embedding')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # 示例数据
    X = np.random.rand(100, 5)
    bounds = [(0, 1), (0, 1),(0,1), (0,1), (0,1)]
    fp = FootPrint(X, bounds)
    fp.calculate_distances()
    fp.get_mds()
    fp.plot_embedding()
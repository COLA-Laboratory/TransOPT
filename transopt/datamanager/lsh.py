import numpy as np
from collections import defaultdict

from transopt.datamanager.minhash import MinHasher

class LSHCache:
    def __init__(self, hasher, num_bands=10):
        """
        Initialize the LSH object with the specified number of bands and rows per band.

        Parameters:
        -----------
        hasher: MinHasher
            An object that computes minhashes for a given input text.

        num_bands: int
            The number of bands to divide the minhash signature matrix into.
        """
        assert (
            hasher.num_hashes % num_bands == 0
        ), "num_hashes must be divisible by num_bands"
        
        self.buckets = [defaultdict(set) for _ in range(num_bands)]
        self.hasher = hasher
        
        self.band_width = hasher.num_hashes // num_bands
        self.num_bands = num_bands
        
        self.fingerprints = {}

    def add(self, key, vector):
        """
        Add a multidimensional vector to the cache.

        Parameters:
        -----------
        key: any hashable
            A unique identifier for the vector.

        vector: tuple (str, str, int, int)
            A tuple representing the multidimensional vector. The tuple format is:
            (task_name, variable_names, num_variables, num_objectives)

        """
        if vector is None:
            return
        # Compute a combined fingerprint for the string dimensions
        combined_fp = []
        for dimension in vector[:2]:  # Only take the first two string dimensions
            combined_fp.extend(self.hasher.fingerprint(dimension))

        # Incorporate the integer dimensions by modifying the bucket key
        num_variables = vector[2]
        num_objectives = vector[3]
        
        # Store the combined fingerprint
        self.fingerprints[key] = (combined_fp, num_variables, num_objectives)
        
        # Divide the fingerprint into bands and store in buckets with the integers as part of the key
        for band_idx in range(self.num_bands):
            start = band_idx * self.band_width
            end = start + self.band_width
            band_fp = (tuple(combined_fp[start:end]), num_variables, num_objectives)
            self.buckets[band_idx][band_fp].add(key)

            
    def query(self, vector):
        """
        Query similar vectors in the cache.

        Parameters:
        -----------
        vector: tuple of (str, str, int, int)
            The multidimensional vector to find similar items to. The format is:
            (task_name, variable_names, num_variables, num_objectives)

        Returns:
        --------
        set
            A set of keys of similar vectors.
        """
        if vector is None:
            return set()
        similar_items = set()
        combined_fp = []
        for dimension in vector[:2]:  # Only take the first two string dimensions
            combined_fp.extend(self.hasher.fingerprint(dimension))

        num_variables = vector[2]
        num_objectives = vector[3]

        # Check for similarity across all bands
        for band_idx in range(self.num_bands):
            start = band_idx * self.band_width
            end = start + self.band_width
            band_fp = (tuple(combined_fp[start:end]), num_variables, num_objectives)
            if band_fp in self.buckets[band_idx]:
                similar_items.update(self.buckets[band_idx][band_fp])

        return similar_items
        
        

if __name__ == "__main__":
    # Example usage assuming MinHasher class is defined and imported correctly.
    hasher = MinHasher(num_hashes=200, char_ngram=2, random_state=42)
    lsh_cache = LSHCache(hasher, num_bands=10)
    lsh_cache.add("doc1", ("parameters1", "objectives1", 10, 5))
    lsh_cache.add("doc2", ("parameters2", "objectives2", 10, 5))
    print(lsh_cache.query(("parameters2", "objectives2", 10, 5)))
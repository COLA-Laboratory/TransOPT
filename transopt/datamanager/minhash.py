from concurrent.futures import ThreadPoolExecutor

import mmh3
import numpy as np


class MinHasher:
    def __init__(self, num_hashes, char_ngram, random_state=None):
        """
        Parameters:
        -----------
        num_hashes: int
            The number of hash functions to use. A minhash is computed for each
            hash function derived from different random seeds.

        char_ngram: int
            The number of consecutive characters to include in a sliding window
            when creating the document shingles.

        random_state: None, int, np.random.RandomState
            A random state to initialise the random number generator with.
        """
        self.num_hashes = num_hashes
        self.char_ngram = char_ngram

        random_state = np.random.RandomState(random_state)
        self._seeds = random_state.randint(0, 1e6, size=num_hashes)

    @property
    def num_seeds(self):
        return len(self._seeds)

    def get_shingles(self, text):
        """Extract character-based shingles from text."""
        return set(
            text[i : i + self.char_ngram]
            for i in range(len(text) - self.char_ngram + 1)
        )

    def fingerprint(self, text):
        shingles = self.get_shingles(text)
        minhashes = [float("inf")] * self.num_hashes
        for shingle in shingles:
            # Ensure the input is in bytes for mmh3
            encoded_shingle = shingle.encode("utf-8")
            for i, seed in enumerate(self._seeds):
                hash_val = mmh3.hash(encoded_shingle, seed) % (2**32)
                if hash_val < minhashes[i]:
                    minhashes[i] = hash_val
        
        return minhashes
    
    def estimate_similarity(self, fp1, fp2):
        return sum(1 for x, y in zip(fp1, fp2) if x == y) / self.num_hashes


def jaccard_similarity(set1, set2):
    if not isinstance(set1, set):
        set1 = set(set1)
    if not isinstance(set2, set):
        set2 = set(set2)
    return len(set1.intersection(set2)) / len(set1.union(set2))


if __name__ == "__main__":
    text1 = "Lorem Ipsum dolor sit ametsdaasdsad"
    text2 = "Lorem Ipsum dolor sit amet is how dummy text starts"

    # Create a MinHasher instance
    hasher = MinHasher(num_hashes=100, char_ngram=2, random_state=12345)

    # Compute shingles for both texts
    shingles1 = hasher.get_shingles(text1)
    shingles2 = hasher.get_shingles(text2)

    # Compute MinHashes for both texts
    fp1 = hasher.fingerprint(text1)
    fp2 = hasher.fingerprint(text2)

    # Comparing MinHash signatures to estimate similarity
    estimated_similarity = hasher.estimate_similarity(fp1, fp2)
    print(f"Estimated similarity: {estimated_similarity:.4f}")
    print(
        f"Jaccard similarity: {jaccard_similarity(hasher.get_shingles(text1), hasher.get_shingles(text2)):.4f}"
    )

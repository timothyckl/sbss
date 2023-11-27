import pytest
import numpy as np
from sbss import SimilarityStratifiedSplit

def sample_sim_func(X):
    # Example similarity function returning random values for each pair of samples
    n_samples = X.shape[0]
    return np.random.rand(n_samples, n_samples)

class TestSimilarityStratifiedSplit:
    @staticmethod
    def test_init():
        splitter = SimilarityStratifiedSplit(n_splits=3, sim_func=sample_sim_func, shuffle=True)
        assert splitter.n_splits == 3
        assert splitter.sim_func == sample_sim_func
        assert splitter.shuffle is True

    @staticmethod
    def test_get_n_splits():
        splitter = SimilarityStratifiedSplit(n_splits=5, sim_func=sample_sim_func)
        assert splitter.get_n_splits() == 5

    @staticmethod
    def test_shuffle_dataset():
        splitter = SimilarityStratifiedSplit(n_splits=2, sim_func=sample_sim_func, shuffle=True)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        shuffled_X, shuffled_y = splitter._shuffle_dataset(X, y)
        assert len(shuffled_X) == len(X)
        assert len(shuffled_y) == len(y)

    @staticmethod
    def test_validate():
        splitter = SimilarityStratifiedSplit(n_splits=4, sim_func=sample_sim_func)
        class_counts = np.array([3, 2])
        min_samples_per_class = 2
        with pytest.raises(ValueError):
            splitter._validate(class_counts, min_samples_per_class)

    @staticmethod
    def test_encode_labels():
        splitter = SimilarityStratifiedSplit(n_splits=2, sim_func=sample_sim_func)
        y = np.array([1, 2, 3, 1, 2, 3])
        num_classes = splitter._encode_labels(y)
        assert num_classes == 3

    @staticmethod
    def test_split():
        splitter = SimilarityStratifiedSplit(n_splits=2, sim_func=sample_sim_func)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        splits = list(splitter.split(X, y))
        assert len(splits) == 2
        for train_indices, test_indices in splits:
            assert len(train_indices) + len(test_indices) == len(X)
            assert len(np.intersect1d(train_indices, test_indices)) == 0

    @staticmethod
    def test_validate_pass():
        splitter = SimilarityStratifiedSplit(n_splits=2, sim_func=sample_sim_func)
        class_counts = np.array([3, 2])
        min_samples_per_class = 2
        splitter._validate(class_counts, min_samples_per_class)
        assert True 

    @staticmethod
    def test_validate_fail():
        splitter = SimilarityStratifiedSplit(n_splits=4, sim_func=sample_sim_func)
        class_counts = np.array([3, 2])
        min_samples_per_class = 2
        with pytest.raises(ValueError):
            splitter._validate(class_counts, min_samples_per_class)
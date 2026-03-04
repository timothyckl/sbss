import pytest
import numpy as np
from scipy.spatial.distance import cdist
from sbss import SimilarityStratifiedSplit


def sample_dist_func(X):
    """Computes pairwise euclidean distances between all samples."""
    return cdist(X, X, metric='euclidean')


class TestSimilarityStratifiedSplit:
    @staticmethod
    def test_init():
        # verify default shuffle is False
        splitter = SimilarityStratifiedSplit(n_splits=3, dist_func=sample_dist_func)
        assert splitter.n_splits == 3
        assert splitter.dist_func == sample_dist_func
        assert splitter.shuffle is False

        # verify explicit shuffle=True is stored correctly
        splitter_shuffled = SimilarityStratifiedSplit(n_splits=3, dist_func=sample_dist_func, shuffle=True)
        assert splitter_shuffled.shuffle is True

    @staticmethod
    def test_get_n_splits():
        splitter = SimilarityStratifiedSplit(n_splits=5, dist_func=sample_dist_func)
        assert splitter.get_n_splits() == 5

    @staticmethod
    def test_shuffle_dataset():
        splitter = SimilarityStratifiedSplit(n_splits=2, dist_func=sample_dist_func, shuffle=True)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        shuffled_X, shuffled_y = splitter._shuffle_dataset(X, y)
        assert len(shuffled_X) == len(X)
        assert len(shuffled_y) == len(y)
        # verify the shuffled arrays contain exactly the same elements in (potentially) different order
        assert np.array_equal(np.sort(shuffled_X, axis=0), np.sort(X, axis=0))
        assert np.array_equal(np.sort(shuffled_y), np.sort(y))

    @staticmethod
    def test_encode_labels():
        splitter = SimilarityStratifiedSplit(n_splits=2, dist_func=sample_dist_func)
        y = np.array([1, 2, 3, 1, 2, 3])
        num_classes, _ = splitter._encode_labels(y)
        assert num_classes == 3

    @staticmethod
    def test_split():
        splitter = SimilarityStratifiedSplit(n_splits=2, dist_func=sample_dist_func)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        splits = list(splitter.split(X, y))
        assert len(splits) == 2
        for train_indices, test_indices in splits:
            assert len(train_indices) + len(test_indices) == len(X)
            assert len(np.intersect1d(train_indices, test_indices)) == 0

        # verify label encoding fix works for non-zero-indexed labels
        y_non_zero = np.array([1, 2, 1, 2])
        X_non_zero = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        splits_non_zero = list(splitter.split(X_non_zero, y_non_zero))
        assert len(splits_non_zero) == 2
        for train_indices, test_indices in splits_non_zero:
            assert len(train_indices) + len(test_indices) == len(X_non_zero)
            assert len(np.intersect1d(train_indices, test_indices)) == 0

    @staticmethod
    def test_validate_pass():
        splitter = SimilarityStratifiedSplit(n_splits=2, dist_func=sample_dist_func)
        class_counts = np.array([3, 2])
        min_samples_per_class = 2
        splitter._validate(class_counts, min_samples_per_class)
        assert True

    @staticmethod
    def test_validate_fail():
        # n_splits exceeds all class counts — both np.all and np.any raise
        splitter = SimilarityStratifiedSplit(n_splits=4, dist_func=sample_dist_func)
        class_counts = np.array([3, 2])
        min_samples_per_class = 2
        with pytest.raises(ValueError):
            splitter._validate(class_counts, min_samples_per_class)

        # n_splits exceeds only one class count — np.all would not raise but np.any correctly does
        splitter2 = SimilarityStratifiedSplit(n_splits=4, dist_func=sample_dist_func)
        class_counts2 = np.array([5, 2])
        min_samples_per_class2 = 2
        with pytest.raises(ValueError):
            splitter2._validate(class_counts2, min_samples_per_class2)

    @staticmethod
    def test_split_with_shuffle():
        splitter = SimilarityStratifiedSplit(n_splits=2, dist_func=sample_dist_func, shuffle=True)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        splits = list(splitter.split(X, y))
        assert len(splits) == 2
        # verify no overlap between train and test within each split
        for train_indices, test_indices in splits:
            assert len(np.intersect1d(train_indices, test_indices)) == 0
        # verify all indices are covered across test splits
        all_test_indices = np.concatenate([test for _, test in splits])
        assert len(np.unique(all_test_indices)) == len(X)

    @staticmethod
    def test_get_n_splits_accepts_sklearn_kwargs():
        # verify sklearn-style keyword arguments are accepted without error
        splitter = SimilarityStratifiedSplit(n_splits=3, dist_func=sample_dist_func)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y = np.array([0, 1, 0, 1, 0, 1])
        assert splitter.get_n_splits(X=X, y=y, groups=None) == 3
        # verify zero-arg form still works as before
        assert splitter.get_n_splits() == 3

    @staticmethod
    def test_split_accepts_groups_kwarg():
        # verify groups=None keyword argument is accepted without error
        splitter = SimilarityStratifiedSplit(n_splits=2, dist_func=sample_dist_func)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        splits = list(splitter.split(X, y, groups=None))
        assert len(splits) == 2
        for train_indices, test_indices in splits:
            assert len(np.intersect1d(train_indices, test_indices)) == 0

    @staticmethod
    def test_split_raises_on_none_y():
        # verify a clear error is raised when y is omitted (as required by sbss)
        splitter = SimilarityStratifiedSplit(n_splits=2, dist_func=sample_dist_func)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        with pytest.raises(ValueError, match="y is required by the SBSS algorithm"):
            next(splitter.split(X))

    @staticmethod
    def test_cross_val_score_integration():
        sklearn_model_selection = pytest.importorskip("sklearn.model_selection")
        sklearn_dummy = pytest.importorskip("sklearn.dummy")

        cross_val_score = sklearn_model_selection.cross_val_score
        DummyClassifier = sklearn_dummy.DummyClassifier

        n_splits = 2
        splitter = SimilarityStratifiedSplit(n_splits=n_splits, dist_func=sample_dist_func)
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])

        # verify cross_val_score runs end-to-end with our splitter
        scores = cross_val_score(DummyClassifier(), X, y, cv=splitter)
        assert len(scores) == n_splits

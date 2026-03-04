import warnings
import numpy as np
from typing import Callable
from numpy.typing import NDArray

class SimilarityStratifiedSplit:
  """
  SBSS (Similarity Based Stratified Splitting: https://arxiv.org/abs/2010.06099) considers both
  input and output space to create splits, unlike conventional stratified methods that focus solely
  on output distribution. By grouping similar samples within the same label into separate splits,
  SBSS ensures balanced partitions covering diverse dataset regions while maintaining approximately
  equal input and output distribution across all splits.

  Parameters
  ----------
  n_splits : int
      Number of splits to generate.
  dist_func : callable
      Function to compute distances between samples.
  shuffle : bool, optional
      Whether to shuffle the dataset before splitting. Default is False.

  Notes
  -----
  This class is compatible with sklearn's cross-validation utilities (e.g. ``cross_val_score``,
  ``GridSearchCV``). The ``get_n_splits`` and ``split`` methods accept the standard sklearn
  ``X``, ``y``, and ``groups`` keyword arguments, though ``groups`` is ignored by the algorithm.
  """
  def __init__(self, n_splits: int, dist_func: Callable, shuffle: bool = False) -> None:
    self.n_splits = n_splits
    self.dist_func = dist_func
    self.shuffle = shuffle

  def get_n_splits(self, X=None, y=None, groups=None) -> int:
    """
    Returns the number of splitting iterations for cross-validation.

    Parameters
    ----------
    X : ignored
        Not used, present for sklearn API compatibility.
    y : ignored
        Not used, present for sklearn API compatibility.
    groups : ignored
        Not used, present for sklearn API compatibility.

    Returns
    -------
    int
        The number of splits.
    """
    return self.n_splits

  def _validate(self, class_counts: NDArray, min_samples_per_class: int):
    """
    Validates parameters for stratified splitting.

    Parameters
    ----------
    class_counts : array-like
        Number of samples for each class.
    min_samples_per_class : int
        Minimum number of samples per class.

    Raises
    ----------
    ValueError
        If the number of folds is greater than the number of members in any class.
    Warning
        If the least populated class has fewer samples than the specified number of folds.
    """
    if np.any(self.n_splits > class_counts):
        raise ValueError("Number of folds cannot be greater than the number of members in each class.")

    if self.n_splits > min_samples_per_class:
        warnings.warn(f"The least populated class in labels has only {min_samples_per_class} members, "
                      f"which is less than the specified number of folds: {self.n_splits}")

  def _encode_labels(self, y: NDArray) -> tuple[int, NDArray]:
    """
    Encodes the labels in 'y' ensuring classes are indexed by order of first appearance.
    It calculates the number of unique classes, the count of samples per class, and validates
    the encoded classes' distribution.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target labels to be encoded.

    Returns
    ----------
    num_classes : int
        The number of unique classes in 'y' after encoding.
    encoded_labels : NDArray
        The encoded labels array with classes indexed by order of first appearance.
    """
    # default checking from scikitlearn kfold
    _, labels_idx, labels_inverse = np.unique(y, return_index=True, return_inverse=True)
    _, class_permutation = np.unique(labels_idx, return_inverse=True)
    encoded_labels = class_permutation[labels_inverse]

    num_classes = len(labels_idx)
    class_counts = np.bincount(encoded_labels)
    min_samples_per_class = np.min(class_counts)

    self._validate(class_counts, min_samples_per_class)

    return num_classes, encoded_labels

  def _shuffle_dataset(self, X: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    """
    Shuffles the dataset by applying a random permutation to both X and y.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input features.
    y : array-like of shape (n_samples,)
        Target labels.

    Returns
    ----------
    X_shuffled : NDArray
        Shuffled input features.
    y_shuffled : NDArray
        Shuffled target labels.
    """
    # generate a single permutation and apply it consistently to both arrays
    permutation = np.random.permutation(len(y))
    return X[permutation], y[permutation]

  def split(self, X: NDArray, y: NDArray = None, groups=None):
    """
    Generate indices to split data into training and test set.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where `n_samples` is the number of samples
        and `n_features` is the number of features.
    y : array-like of shape (n_samples,)
        The target variable for supervised learning problems.
    groups : ignored
        Not used, present for sklearn API compatibility.

    Yields
    ------
    train_indices : ndarray
        The training set indices for that split.
    test_indices : ndarray
        The testing set indices for that split.

    Raises
    ------
    ValueError
        If ``y`` is ``None``, since target labels are required by the SBSS algorithm.
    """
    if y is None:
        raise ValueError(
            "y is required by the SBSS algorithm and cannot be None. "
            "Provide target labels as a 1-D array of shape (n_samples,)."
        )

    # shuffle dataset before processing if requested
    if self.shuffle is True:
      X, y = self._shuffle_dataset(X, y)

    num_classes, encoded_labels = self._encode_labels(y)
    distances = self.dist_func(X)

    # boolean array to track used sample indices
    used_indices = np.zeros(len(y)).astype(bool)
    folds_list = [[] for _ in range(self.n_splits)]

    for class_label in range(num_classes):
      # get indices for current class using encoded labels to support non-zero-indexed labels
      class_indices = encoded_labels == class_label
      samples_to_split = class_indices.sum()

      # iterate while there are enough samples in the class to split into folds
      while samples_to_split >= self.n_splits:
        # identify unused samples in current class and get pivot sample
        considered_indices = (~used_indices) & class_indices
        sum_distances = np.nansum(distances[:, considered_indices], axis=1)
        sum_distances[~considered_indices] = np.inf
        # smallest distance sum is chosen to be pivot sample
        pivot_idx = np.argpartition(sum_distances, 0)[0]

        used_indices[pivot_idx] = True
        # exclude pivot from consideration via the same mechanism as other selected samples
        considered_indices[pivot_idx] = False
        nearby_samples = [pivot_idx]

        # find N-1 similar samples
        for fold_idx in range(1, self.n_splits):
          sum_distances = np.nansum(distances[:, nearby_samples], axis=1)
          sum_distances[~considered_indices] = np.inf
          sum_distances[pivot_idx] = np.inf

          closest_sample_idx = np.argpartition(sum_distances, 0)[0]
          nearby_samples.append(closest_sample_idx)

          # mark index in mask arrays as used and considered
          used_indices[closest_sample_idx] = True
          considered_indices[closest_sample_idx] = False

        # shuffle for stochasticity when appending to splits
        np.random.shuffle(nearby_samples)

        for fold_idx in range(self.n_splits):
          folds_list[fold_idx].append(nearby_samples[fold_idx])

        # decrement samples_to_split after filling folds
        samples_to_split -= self.n_splits

    folds = np.array(folds_list)

    for fold_idx in range(self.n_splits):
      train_splits = np.ones(self.n_splits).astype(bool)
      train_splits[fold_idx] = False

      test_indices = folds[fold_idx]
      train_indices = folds[train_splits, :].ravel()

      yield train_indices, test_indices

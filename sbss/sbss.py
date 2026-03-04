import warnings
import numpy as np
from collections.abc import Iterator
from typing import Callable
from numpy.typing import NDArray

class SimilarityStratifiedSplit:
  """SBSS (Similarity Based Stratified Splitting: https://arxiv.org/abs/2010.06099) considers both
  input and output space to create splits, unlike conventional stratified methods that focus solely
  on output distribution. By grouping similar samples within the same label into separate splits,
  SBSS ensures balanced partitions covering diverse dataset regions while maintaining approximately
  equal input and output distribution across all splits.

  Args:
      n_splits: Number of splits to generate.
      dist_func: Function to compute pairwise distances between samples.
      shuffle: Whether to shuffle the dataset before splitting. Defaults to False.

  Note:
      Compatible with sklearn's cross-validation utilities (e.g. ``cross_val_score``,
      ``GridSearchCV``). The ``get_n_splits`` and ``split`` methods accept the standard
      sklearn ``X``, ``y``, and ``groups`` keyword arguments, though ``groups`` is ignored.
  """
  def __init__(self, n_splits: int, dist_func: Callable[[NDArray], NDArray], shuffle: bool = False) -> None:
    self.n_splits = n_splits
    self.dist_func = dist_func
    self.shuffle = shuffle

  def get_n_splits(self, X=None, y=None, groups=None) -> int:
    """Returns the number of splitting iterations for cross-validation.

    Args:
        X: Ignored. Present for sklearn API compatibility.
        y: Ignored. Present for sklearn API compatibility.
        groups: Ignored. Present for sklearn API compatibility.

    Returns:
        Number of splits.
    """
    return self.n_splits

  def _validate(self, class_counts: NDArray, min_samples_per_class: int) -> None:
    """Validates parameters for stratified splitting.

    Args:
        class_counts: Number of samples for each class.
        min_samples_per_class: Minimum number of samples per class.

    Raises:
        ValueError: If n_splits exceeds the number of members in any class.

    Note:
        Emits a UserWarning if the least populated class has fewer samples
        than n_splits.
    """
    if np.any(self.n_splits > class_counts):
        raise ValueError("Number of folds cannot be greater than the number of members in each class.")

    if self.n_splits > min_samples_per_class:
        warnings.warn(f"The least populated class in labels has only {min_samples_per_class} members, "
                      f"which is less than the specified number of folds: {self.n_splits}")

  def _encode_labels(self, y: NDArray) -> tuple[int, NDArray]:
    """Encodes labels in y ensuring classes are indexed by order of first appearance.

    Calculates the number of unique classes, the count of samples per class, and
    validates the encoded classes' distribution.

    Args:
        y: Target labels of shape (n_samples,).

    Returns:
        Tuple of (num_classes, encoded_labels) where num_classes is the number of
        unique classes and encoded_labels is the array with classes re-indexed by
        order of first appearance.
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
    """Shuffles the dataset by applying a random permutation to both X and y.

    Args:
        X: Input features of shape (n_samples, n_features).
        y: Target labels of shape (n_samples,).

    Returns:
        Tuple of (X_shuffled, y_shuffled) with a consistent permutation applied.
    """
    # generate a single permutation and apply it consistently to both arrays
    permutation = np.random.permutation(len(y))
    return X[permutation], y[permutation]

  def split(self, X: NDArray, y: NDArray = None, groups=None) -> Iterator[tuple[NDArray, NDArray]]:
    """Generates indices to split data into training and test sets.

    Args:
        X: Input features of shape (n_samples, n_features).
        y: Target labels of shape (n_samples,).
        groups: Ignored. Present for sklearn API compatibility.

    Yields:
        Tuple of (train_indices, test_indices) as 1-D integer arrays.

    Raises:
        ValueError: If y is None, since target labels are required by the SBSS algorithm.
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

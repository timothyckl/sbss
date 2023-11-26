"""
todo: document algorithm
"""

import warnings
import numpy as np

class SimilarityStratifiedSplit:
  """
  todo: document
  """
  def __init__(self, n_splits, sim_func, shuffle=False):
    self.n_splits = n_splits
    self.sim_func = sim_func
    self.shuffle = shuffle
    self.shuffled_dataset = None  # used to store shuffled dataset for interaction after split operation

  def __str__(self):
    return f"SimilarityStratifiedSplit(n_splits={self.n_splits}, sim_func={self.sim_func}, shuffle={self.shuffle})"

  def get_n_splits(self):
    """
    todo: document
    """
    return self.n_splits

  def _shuffle_dataset(self, X, y):
    """
    todo: document
    """
    permutation = np.random.permutation(len(X))
    shuffled_x = X[permutation]
    shuffled_y = y[permutation]
    
    self.shuffled_dataset = (shuffled_x, shuffled_y)

    return self.shuffled_dataset

  def _validate(self, class_counts, min_samples_per_class):
    """
    todo: document
    """
    if np.all(self.n_splits > class_counts):
        raise ValueError("Number of folds cannot be greater than the number of members in each class.")

    if self.n_splits > min_samples_per_class:
        warnings.warn(f"The least populated class in labels has only {min_samples_per_class} members, "
                      f"which is less than the specified number of folds: {self.n_splits}")

  def _encode_labels(self, y):
    """
    todo: document
    """
    _, labels_idx, labels_inverse = np.unique(y, return_index=True, return_inverse=True)
    _, class_permutation = np.unique(labels_idx, return_inverse=True)
    encoded_labels = class_permutation[labels_inverse]

    num_classes = len(labels_idx)
    class_counts = np.bincount(encoded_labels)
    min_samples_per_class = np.min(class_counts)

    self._validate(class_counts, min_samples_per_class)

    return num_classes

  def split(self, X, y):
    """
    todo: document
    """
    if self.shuffle:
      X, y = self._shuffle_dataset(X, y)

    num_classes = self._encode_labels(y)

    train_idx = None
    test_idx = None

    distances = self.sim_func(X)

    used_indices = np.zeros(len(y)).astype(bool)
    folds_list = [[] for _ in range(self.n_splits)]
    fold_label_column = []

    for class_label in range(num_classes):
      class_indices = y.squeeze() == class_label
      samples_to_split = class_indices.sum()

      while samples_to_split >= self.n_splits:
        considered_indices = (~used_indices) & class_indices

        sum_distances = np.sum(distances[:, considered_indices], axis=1)
        sum_distances[~considered_indices] = np.inf
        pivot_idx = np.argpartition(sum_distances, 0)[0]

        used_indices[pivot_idx] = True
        nearby_samples = [pivot_idx]

        for fold_idx in range(1, self.n_splits):
          sum_distances = np.sum(distances[:, nearby_samples], axis=1)
          sum_distances[~considered_indices] = np.inf
          sum_distances[pivot_idx] = np.inf

          closest_sample_idx = np.argpartition(sum_distances, 0)[0]
          nearby_samples.append(closest_sample_idx)

          used_indices[closest_sample_idx] = True
          considered_indices[closest_sample_idx] = False

        fold_label_column.append(class_label)
        np.random.shuffle(nearby_samples)

        for fold_idx in range(self.n_splits):
          folds_list[fold_idx].append(nearby_samples[fold_idx])

        samples_to_split -= self.n_splits

    folds = np.array(folds_list)
    fold_label_column = np.array(fold_label_column)

    for fold_idx in range(self.n_splits):
      train_splits = np.ones(self.n_splits).astype(bool)
      train_splits[fold_idx] = False

      test_indices = folds[fold_idx]
      train_indices = folds[train_splits, :].ravel()

      yield train_indices, test_indices

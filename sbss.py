import warnings
import numpy as np

class SimilarityStratifiedSplit(BaseSplitter):
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
  sim_func : callable
      Function to compute similarity between samples.
  shuffle : bool, default=False
      Whether to shuffle the dataset before splitting.
  """
  def __init__(self, n_splits, sim_func, shuffle=False):
    self.n_splits = n_splits
    self.sim_func = sim_func
    self.shuffle = shuffle
    self.shuffled_dataset = None

  def get_n_splits(self):
    """Returns the number of splitting iterations for cross-validation"""
    return self.n_splits

  def _shuffle_dataset(self, X, y):
    """
    Shuffles the input features and corresponding labels using a random permutation.
    
    Parameters
    ----------
    X : array-like
        Input features.
    y : array-like
        Corresponding labels.
        
    Returns
    ----------
    tuple
        A tuple containing shuffled input features and labels.
    """
    permutation = np.random.permutation(len(X))
    shuffled_x = X[permutation]
    shuffled_y = y[permutation]

    self.shuffled_dataset = (shuffled_x, shuffled_y)

    return self.shuffled_dataset

  def _validate(self, class_counts, min_samples_per_class):
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
        If the number of folds is greater than the number of members in each class.
    Warning
        If the least populated class has fewer samples than the specified number of folds.
    """
    if np.all(self.n_splits > class_counts):
        raise ValueError("Number of folds cannot be greater than the number of members in each class.")

    if self.n_splits > min_samples_per_class:
        warnings.warn(f"The least populated class in labels has only {min_samples_per_class} members, "
                      f"which is less than the specified number of folds: {self.n_splits}")

  def _encode_labels(self, y):
    """
    Encodes the labels in 'y' based on lexicographic order, ensuring classes are encoded by order of appearance.
    It calculates the number of unique classes, the count of samples per class, and validates the encoded classes' distribution.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target labels to be encoded.
        
    Returns
    ----------
    num_classes: int
                 The number of unique classes in 'y' after encoding.
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
    Generate indices to split data into training and test set.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where `n_samples` is the number of samples
        and `n_features` is the number of features.

    y : array-like of shape (n_samples,)
        The target variable for supervised learning problems.

    Yields
    ------
    train : ndarray
      The training set indices for that split.

    test : ndarray
          The testing set indices for that split.
    """
    if self.shuffle:
      X, y = self._shuffle_dataset(X, y)

    num_classes = self._encode_labels(y)
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

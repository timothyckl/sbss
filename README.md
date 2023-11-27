# Similarity Stratified Split

Implementation of the Similarity-Based Stratified Splitting algorithm described in [Similarity Based Stratified Splitting: an approach to train better classifiers](https://arxiv.org/abs/2010.06099).

## Overview

WIP

## Install

PyPI
```bash
pip install sbss
```

Local install
```
git clone https://github.com/timothyckl/similarity-stratified-split.git
cd ./similarity-stratified-split
pip install -e .
```

## Usage

```python
from scipy.spatial import distance
from sbss import SimilarityStratifiedSplit

def get_distances(x):
    distances = distance.squareform(distance.pdist(x, metric='euclidean'))
    return distances

# inputs are recommended to be normalized
X = np.random.rand(1000, 128)
y = np.random.randint(0, 1000, (20,))

n_splits = 3
s = SimilarityStratifiedSplit(n_splits, get_distances, shuffle=True)

for train_index, test_index in s.split(data, labels):
  print(f"Train indices: {train_index}\nTest indices: {test_index}")
  print("="*100)
```

## References

- Farias, F., Ludermir, T. and Bastos-Filho, C. (2020) Similarity based stratified splitting: An approach to train better classifiers, arXiv.org. Available at: https://arxiv.org/abs/2010.06099 (Accessed: 27 November 2023). 
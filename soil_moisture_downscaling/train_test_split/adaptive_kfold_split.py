# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause
# Reference: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_split.py

import numpy as np
from random import randint
from sklearn.utils.validation import _num_samples


class AdaptiveKFold:
    def __init__(self, n_splits, fold_size):
        self.n_splits = n_splits
        self.fold_size = fold_size

    def split(self, X, y=None, groups=None):
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    def _iter_test_masks(self, X=None, y=None, groups=None):
        for test_index in self._iter_test_indices(X):
            test_mask = np.zeros(_num_samples(X), dtype=np.bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        n_splits = self.n_splits
        fold_size = self.fold_size

        for i in range(n_splits):
            if i == 0:
                yield np.append(indices[:fold_size], indices[n_samples - fold_size:])
            else:
                start = randint(0, n_samples-fold_size)
                second_start = randint(0, n_samples-fold_size)
                while start <= second_start < start+fold_size:
                    second_start = randint(0, n_samples-fold_size)
                yield np.append(indices[start:start+fold_size], indices[second_start:second_start+fold_size])

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

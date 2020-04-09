from sklearn.model_selection import StratifiedKFold

import numpy as np


class BinnedStratifiedKFold(StratifiedKFold):
    """
    StratifiedKFold doesn't support continuous variables yet:
    https://github.com/scikit-learn/scikit-learn/issues/4757

    PR is in progress:
    https://github.com/scikit-learn/scikit-learn/pull/14560/files
    """

    def __init__(self, n_splits=5, shuffle=False, n_bins=5,
                 random_state=None):
        super().__init__(n_splits, shuffle, random_state)
        self.n_bins = n_bins
        if n_bins < 2:
            raise ValueError("Need at least two bins, got {}.".format(
                n_bins))

    @staticmethod
    def get_bins(y, n_bins):
        percentiles = np.percentile(y, np.linspace(0, 100, n_bins))
        return np.searchsorted(percentiles[1:-1], y)

    def _make_test_folds(self, X, y):
        bins = self.get_bins(y, self.n_bins + 1)
        return super()._make_test_folds(X, bins)

    def _iter_test_indices(self, X=None, y=None, groups=None):
        pass

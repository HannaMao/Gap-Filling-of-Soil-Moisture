# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from scipy.stats.stats import pearsonr


def pearson_corr_as_scorer(y, y_hat):
    corr = pearsonr(y_hat, y)
    return corr[0]


__all__ = ["pearson_corr_as_scorer"]


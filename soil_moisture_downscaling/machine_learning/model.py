# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from .utils import pre_process

from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats.stats import pearsonr
import numpy as np


class MLModel(object):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def apply_model(self, features_names):
        pass

    def apply_predict(self, method, oob=False):
        if isinstance(method, RandomForestRegressor) and oob:
            print('Oob score for training dataset:', method.oob_score_)

        train_prediction = method.predict(self.X_train)
        test_prediction = method.predict(self.X_test)

        train_r2 = r2_score(self.y_train, train_prediction)
        test_r2 = r2_score(self.y_test, test_prediction)
        train_corr = pearsonr(self.y_train.ravel(), train_prediction.ravel())
        test_corr = pearsonr(self.y_test.ravel(), test_prediction.ravel())
        train_rmse = sqrt(mean_squared_error(self.y_train, train_prediction))
        test_rmse = sqrt(mean_squared_error(self.y_test, test_prediction))
        train_ubrmse = np.sqrt(np.mean(((train_prediction - np.mean(train_prediction))
                                        - (self.y_train - np.mean(self.y_train)))**2))
        test_ubrmse = np.sqrt(np.mean(((test_prediction - np.mean(test_prediction))
                                       - (self.y_test - np.mean(self.y_test)))**2))
        train_bias = train_prediction.mean() - self.y_train.mean()
        test_bias = test_prediction.mean() - self.y_test.mean()
        print('R2 score for train dataset:', train_r2)
        print('R2 score for test dataset:', test_r2)
        print('Correlation for train dataset:', train_corr)
        print('Correlation for test dataset:', test_corr)
        print('RMSE for train dataset:', train_rmse)
        print('RMSE for test dataset:', test_rmse)
        print('ubRMSE for train dataset:', train_ubrmse)
        print('ubRMSE for test dataset:', test_ubrmse)
        print('Bias for train dataset:', train_bias)
        print('Bias for test dataset:', test_bias)

        return train_prediction, test_prediction






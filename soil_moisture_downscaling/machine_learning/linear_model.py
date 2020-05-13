# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from .model import MLModel
from ..metrics import pearson_corr_as_scorer
from ..train_test_split import AdaptiveKFold

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from scipy.stats import randint
from scipy.stats import uniform


def rmse(y, y_pred):
    return np.sqrt(np.mean((y_pred - y)**2))


class LinearModel(MLModel):
    def __init__(self, X_train, y_train, X_test, y_test, seed=192, param_dist=None):
        super(LinearModel, self).__init__(X_train, y_train, X_test, y_test)
        self.param_dist = param_dist
        self.seed = seed

    def _get_cv(self):
        cv_type = self.cv_type

        if cv_type == "kfold":
            return 10
        elif cv_type == "shuffle":
            return KFold(n_splits=10, shuffle=True, random_state=self.seed)
        elif cv_type == "adaptive":
            return AdaptiveKFold(n_splits=10, fold_size=self.fold_size)

    def _get_scorer(self):
        scorer = self.scorer
        search_type = self.search_type

        if scorer == "r2":
            return "r2"
        elif scorer == "corr":
            return make_scorer(pearson_corr_as_scorer)
        elif scorer == "rmse":
            if search_type == "random":
                return "neg_mean_squared_error"

    def apply_model(self, feature_names, cv_type=None, search_type=None, scorer=None):
        self.cv_type = cv_type
        self.search_type = search_type
        self.scorer = scorer

        if cv_type is None:
            lm = LinearRegression(n_jobs=-1)
            lm.fit(self.X_train, self.y_train)
        # else:
        #     if search_type == "random":
        #         parameter_grid = self._get_random_parameter_grid()
        #         rf_search = RandomizedSearchCV(estimator=RandomForestRegressor(n_estimators=300, oob_score=True,
        #                                                                        n_jobs=-1, random_state=self.seed),
        #                                        param_distributions=parameter_grid,
        #                                        scoring=self._get_scorer(),
        #                                        n_iter=100,
        #                                        cv=self._get_cv(),
        #                                        iid=False,
        #                                        random_state=self.seed,
        #                                        n_jobs=-1)
        #         rf_search.fit(self.X_train, self.y_train)
        #         rf = rf_search.best_estimator_
        #
        #         mean_test_score = rf_search.cv_results_["mean_test_score"]
        #         std_test_score = rf_search.cv_results_["std_test_score"]
        #         print(max(mean_test_score), std_test_score[np.argmax(mean_test_score)])
        #         print(rf)
        #
        #     else:
        #         raise ValueError("search_type must be random.")

        oob_prediction, test_prediction = self.apply_predict(lm, oob=True)
        return oob_prediction, test_prediction

    def _get_random_parameter_grid(self):
        n_features = self.X_train.shape[1]

        max_features = randint(1, n_features + 1)
        min_samples_split = randint(2, 51)
        min_samples_leaf = randint(1, 51)
        min_weight_fraction_leaf = uniform(0.0, 0.5)
        max_leaf_nodes = randint(10, 1001)

        return {'max_features': max_features,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                "min_weight_fraction_leaf": min_weight_fraction_leaf,
                "max_leaf_nodes": max_leaf_nodes}






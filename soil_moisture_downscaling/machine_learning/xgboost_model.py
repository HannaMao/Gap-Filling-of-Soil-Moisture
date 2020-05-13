# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from .model import MLModel
from ..metrics import pearson_corr_as_scorer
from ..train_test_split import AdaptiveKFold

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from scipy.stats import randint
from scipy.stats import uniform
from smac.configspace import ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC


def rmse(y, y_pred):
    return np.sqrt(np.mean((y_pred - y)**2))


class XGBoostModel(MLModel):
    def __init__(self, X_train, y_train, X_test, y_test, seed=192, param_dist=None):
        super(XGBoostModel, self).__init__(X_train, y_train, X_test, y_test)
        self.param_dist = param_dist
        self.seed = seed
        train_size = X_train.shape[0]
        test_size = X_test.shape[0]
        self.fold_size = int((train_size * test_size) / ((train_size + test_size) * 2))

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
            elif search_type == "smac":
                return make_scorer(rmse, greater_is_better=False)

    def apply_model(self, feature_names, cv_type=None, search_type=None, scorer=None):
        self.cv_type = cv_type
        self.search_type = search_type
        self.scorer = scorer

        if cv_type is None:
            print("Seed:", self.seed)
            x_gb = XGBRegressor(learning_rate=0.1, n_estimators=300, random_state=self.seed)
            x_gb.fit(self.X_train, self.y_train)
        else:
            if search_type == "random":
                parameter_grid = self._get_random_parameter_grid()
                print("Seed:", self.seed)
                x_gb_search = RandomizedSearchCV(estimator=XGBRegressor(n_estimators=300, random_state=self.seed),
                                                 param_distributions=parameter_grid,
                                                 scoring=self._get_scorer(),
                                                 n_iter=150,
                                                 cv=self._get_cv(),
                                                 iid=False,
                                                 random_state=self.seed,
                                                 n_jobs=-1)
                x_gb_search.fit(self.X_train, self.y_train)
                x_gb = x_gb_search.best_estimator_

                mean_test_score = x_gb_search.cv_results_["mean_test_score"]
                std_test_score = x_gb_search.cv_results_["std_test_score"]
                print(max(mean_test_score), std_test_score[np.argmax(mean_test_score)])
                print(x_gb)
            elif search_type == "smac":
                smac = self._get_smac()
                try:
                    incumbent = smac.optimize()
                finally:
                    incumbent = smac.solver.incumbent
                x_gb = XGBRegressor(learning_rate=0.1, n_estimators=300, seed=self.seed,
                                    max_depth=incumbent["max_depth"],
                                    min_child_weight=incumbent["min_child_weight"],
                                    gamma=incumbent["gamma"],
                                    subsample=incumbent["subsample"],
                                    colsample_bytree=incumbent["colsample_bytree"])
                x_gb.fit(self.X_train, self.y_train)
                print(x_gb)
            else:
                raise ValueError("search_type must be either random or smac.")

        train_prediction, test_prediction = self.apply_predict(x_gb)
        return train_prediction, test_prediction

    def _get_random_parameter_grid(self):
        learning_rate = [0.006, 0.01, 0.03, 0.05, 0.1]
        max_depth = [3, 5, 7, 9, 11, 13]
        min_child_weight = [1, 4, 7, 10, 13, 16, 19]
        subsample = [0.5, 0.75, 1.0]
        colsample_bylevel = [0.4, 0.6, 0.8, 1.0]

        return {'learning_rate': learning_rate,
                'max_depth': max_depth,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                'colsample_bylevel': colsample_bylevel}

    def _x_gb_from_cfg(self, cfg, seed):
        xgbr = XGBRegressor(learning_rate=0.1,
                            n_estimators=300,
                            seed=seed,
                            max_depth=cfg["max_depth"],
                            min_child_weight=cfg["min_child_weight"],
                            gamma=cfg["gamma"],
                            subsample=cfg["subsample"],
                            colsample_bytree=cfg["colsample_bytree"])
        cv_score = cross_val_score(xgbr, self.X_train, self.y_train, cv=self._get_cv(), scoring=self._get_scorer())

        return -1 * np.mean(cv_score)

    def _get_cfg(self):
        cs = ConfigurationSpace()
        max_depth = UniformIntegerHyperparameter("max_depth", 3, 16, default_value=3)
        min_child_weight = UniformIntegerHyperparameter("min_child_weight", 1, 15, default_value=1)
        gamma = UniformFloatHyperparameter("gamma", 0.0, 0.4, default_value=0.0)
        subsample = UniformFloatHyperparameter("subsample", 0.6, 0.9, default_value=0.6)
        colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.6, 0.9, default_value=0.6)

        cs.add_hyperparameters([max_depth, min_child_weight, gamma, subsample, colsample_bytree])
        return cs

    def _get_smac(self):
        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative runtime)
                             "runcount-limit": 100,  # maximum number of function evaluations
                             "cs": self._get_cfg(),  # configuration space
                             "deterministic": "true",
                             "memory_limit": 10000,  # adapt this to reasonable value for your hardware，
                             "output_dir": "SMAC"
                             })
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(self.seed),
                    tae_runner=self._x_gb_from_cfg)
        return smac





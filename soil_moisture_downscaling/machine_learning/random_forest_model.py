# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from .model import MLModel
from ..metrics import pearson_corr_as_scorer
from ..train_test_split import AdaptiveKFold

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from scipy.stats import randint
from scipy.stats import uniform
from smac.configspace import ConfigurationSpace
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from collections import defaultdict
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def rmse(y, y_pred):
    return np.sqrt(np.mean((y_pred - y)**2))


class RandomForestModel(MLModel):
    def __init__(self, X_train, y_train, X_test, y_test, seed=192, param_dist=None):
        super(RandomForestModel, self).__init__(X_train, y_train, X_test, y_test)
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
            elif search_type == "smac":
                return make_scorer(rmse, greater_is_better=False)

    def apply_model(self, feature_names, cv_type=None, search_type=None, scorer=None):
        self.cv_type = cv_type
        self.search_type = search_type
        self.scorer = scorer

        if cv_type is None:
            rf = RandomForestRegressor(n_estimators=300, oob_score=True, n_jobs=-1, random_state=self.seed)
            rf.fit(self.X_train, self.y_train)
        else:
            if search_type == "random":
                parameter_grid = self._get_random_parameter_grid()
                rf_search = RandomizedSearchCV(estimator=RandomForestRegressor(n_estimators=300, oob_score=True,
                                                                               n_jobs=-1, random_state=self.seed),
                                               param_distributions=parameter_grid,
                                               scoring=self._get_scorer(),
                                               n_iter=100,
                                               cv=self._get_cv(),
                                               iid=False,
                                               random_state=self.seed,
                                               n_jobs=-1)
                rf_search.fit(self.X_train, self.y_train)
                rf = rf_search.best_estimator_

                mean_test_score = rf_search.cv_results_["mean_test_score"]
                std_test_score = rf_search.cv_results_["std_test_score"]
                print(max(mean_test_score), std_test_score[np.argmax(mean_test_score)])
                print(rf)

            elif search_type == "smac":
                smac = self._get_smac()
                try:
                    incumbent = smac.optimize()
                finally:
                    incumbent = smac.solver.incumbent
                rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, oob_score=True, random_state=self.seed,
                                           max_features=incumbent["max_features"],
                                           min_samples_split=incumbent["min_samples_split"],
                                           min_samples_leaf=incumbent["min_samples_leaf"],
                                           min_weight_fraction_leaf=incumbent["min_weight_fraction_leaf"],
                                           max_leaf_nodes=incumbent["max_leaf_nodes"])
                rf.fit(self.X_train, self.y_train)
                print(rf)
            else:
                raise ValueError("search_type must be either random or smac.")

        oob_prediction, test_prediction = self.apply_predict(rf, oob=True)
        self.mean_decrease_impurity(rf, feature_names)
        # self.mean_decrease_accuracy(feature_names)
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

    def _rf_from_cfg(self, cfg, seed):
        rfr = RandomForestRegressor(n_estimators=300,
                                    oob_score=True,
                                    n_jobs=-1,
                                    random_state=seed,
                                    max_features=cfg["max_features"],
                                    min_samples_split=cfg["min_samples_split"],
                                    min_samples_leaf=cfg["min_samples_leaf"],
                                    min_weight_fraction_leaf=cfg["min_weight_fraction_leaf"],
                                    max_leaf_nodes=cfg["max_leaf_nodes"])
        cv_score = cross_val_score(rfr, self.X_train, self.y_train, cv=self._get_cv(), scoring=self._get_scorer())

        return -1 * np.mean(cv_score)

    def _get_cfg(self):
        n_features = self.X_train.shape[1]

        cs = ConfigurationSpace()
        max_features = UniformIntegerHyperparameter("max_features", 1, n_features, default_value=1)
        min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 50, default_value=2)
        min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 50, default_value=1)
        min_weight_fraction_leaf = UniformFloatHyperparameter("min_weight_fraction_leaf", 0.0, 0.5, default_value=0.0)
        max_leaf_nodes = UniformIntegerHyperparameter("max_leaf_nodes", 10, 1000, default_value=100)

        cs.add_hyperparameters([max_features, min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes])
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
                    tae_runner=self._rf_from_cfg)
        return smac

    def mean_decrease_accuracy(self, feature_names):
        X_train, y_train = self.X_train, self.y_train
        X_test, y_test = self.X_test, self.y_test
        print('Feature importance - mean_decrease_accuracy:')
        scores = defaultdict(list)

        rf = RandomForestRegressor(n_estimators=20, n_jobs=-1, bootstrap=True)
        rf.fit(X_train, y_train)
        acc = r2_score(y_test, rf.predict(X_test))
        for i in range(X_test.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(y_test, rf.predict(X_t))
            scores[feature_names[i]].append((acc - shuff_acc) / acc)

        accu_lis = [(feature, round(sum(score)/len(score), 4)) for feature, score in scores.items()]
        for index, (feature, score) in enumerate(sorted(accu_lis, key=lambda x: x[1])[::-1]):
            print("%d. %s (%f)" % (index + 1, feature, score))

    def mean_decrease_impurity(self, rf, feature_names, plot=False):
        X_train = self.X_train
        # http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
        importances = rf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print('Feature importance - mean_decrease_impurity:')

        for f in range(X_train.shape[1]):
            print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

        if plot:
            indices = indices[::-1]
            # Plot the feature importances of the forest
            plt.figure()
            plt.title("Feature importances")
            plt.barh(range(X_train.shape[1]), importances[indices],
                     color="r", yerr=std[indices], align="center")
            plt.yticks(range(X_train.shape[1]), [feature_names[indice] for indice in indices])
            plt.ylim([-1, X_train.shape[1]])
            plt.show()






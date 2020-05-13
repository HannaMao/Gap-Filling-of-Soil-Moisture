# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from .random_forest_model import RandomForestModel
from .gradient_boosting_model import GradientBoostingModel
from .xgboost_model import XGBoostModel
from ..utils import find_index, Logger

import os
import sys
import pandas as pd
import numpy as np
import numpy.ma as ma
import csv
from netCDF4 import Dataset
from math import sqrt
from collections import defaultdict
from operator import and_
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats.stats import pearsonr


class ApplyML_Time(object):
    def __init__(self, in_path, f_name, y_name, key_x_name, model, out_path, seeds,
                 selected_features=None, verbose=True, log=True):
        self.in_path = in_path
        self.out_path = out_path
        self.f_name = f_name

        self.log = log
        if self.log:
            log_file = os.path.join(self.out_path, "_".join([self.f_name, model]) + '.txt')
            sys.stdout = Logger(log_file)

        train_file = self.f_name + '_train.csv'
        test_file = self.f_name + '_test.csv'

        df_train = pd.read_csv(os.path.join(in_path, train_file))
        df_test = pd.read_csv(os.path.join(in_path, test_file))

        self.model = model
        self.X_dim_train = np.array(df_train[["time", "lat", "lon"]])
        self.X_dim_test = np.array(df_test[["time", "lat", "lon"]])

        self.y_train = np.array(df_train.pop(y_name))
        self.y_test = np.array(df_test.pop(y_name))
        self.X_train = np.array(df_train[selected_features])
        self.X_test = np.array(df_test[selected_features])
        self.feature_names = selected_features
        self.y_name = y_name
        self.key_x_name = key_x_name
        self.n_tuple = self.X_train.shape[0] + self.X_test.shape[0]
        self.n_test_tuple = self.X_test.shape[0]
        self.seeds = seeds
        self.verbose = verbose
        self.res = dict()
        self.train_predict = None
        self.test_predict = None

        if self.verbose:
            print("File name:", in_path, f_name, ", target variable", y_name + ",", len(self.feature_names), \
                "Selected features:", self.feature_names, "key_x:", key_x_name)
            print("Machine Learning model:", self.model)
            print("Number of tuples:", self.n_tuple)
            print("Number of test tuples:", self.n_test_tuple)
            print("Before applying ML: ")

            self.res["File"] = self.f_name
            self.res["Number_train"] = self.n_tuple - self.n_test_tuple
            self.res["Number_test"] = self.n_test_tuple
            self.res["R2_train_before"] = r2_score(self.y_train, np.array(df_train[key_x_name]))
            self.res["Corr_train_before"] = pearsonr(self.y_train, np.array(df_train[key_x_name]))[0]
            self.res["RMSE_train_before"] = sqrt(mean_squared_error(self.y_train,
                                                                    np.array(df_train[key_x_name])))
            self.res["R2_test_before"] = r2_score(self.y_test, np.array(df_test[key_x_name]))
            self.res["Corr_test_before"] = pearsonr(self.y_test, np.array(df_test[key_x_name]))[0]
            self.res["RMSE_test_before"] = sqrt(mean_squared_error(self.y_test,
                                                                   np.array(df_test[key_x_name])))

            print("R2 for train dataset:", self.res["R2_train_before"])
            print("R2 for test dataset:", self.res["R2_test_before"])
            print("Corr for train dataset:", self.res["Corr_train_before"])
            print("Corr for test dataset", self.res["Corr_test_before"])
            print("RMSE for train dataset:", self.res["RMSE_train_before"])
            print("RMSE for test dataset:", self.res["RMSE_test_before"])

    def clean_up(self):
        if self.log:
            sys.stdout.close()
            sys.stdout = sys.__stdout__

    def apply(self, cv_type=None, search_type=None, scorer=None, param_dist=None):
        model = self.model
        statistics = defaultdict(list)
        predict = {}

        for seed_index in range(len(self.seeds)):
            print("=========" + "ITERATION " + str(seed_index) + "=====================================================")

            ml_model = globals()[model](X_train=self.X_train,
                                        y_train=self.y_train,
                                        X_test=self.X_test,
                                        y_test=self.y_test,
                                        seed=self.seeds[seed_index],
                                        param_dist=param_dist)
            train_predict, test_predict = ml_model.apply_model(cv_type=cv_type,
                                                               search_type=search_type,
                                                               scorer=scorer)
            r2_test = r2_score(self.y_test, test_predict)
            corr_test = pearsonr(self.y_test, test_predict)[0]
            rmse_test = sqrt(mean_squared_error(self.y_test, test_predict))
            predict[corr_test] = {"train_predict": train_predict, "test_predict": test_predict}
            statistics["r2"].append(r2_test)
            statistics["corr"].append(corr_test)
            statistics["rmse"].append(rmse_test)

        best_predict = predict[max(predict.keys())]
        self.train_predict = best_predict["train_predict"]
        self.test_predict = best_predict["test_predict"]

        self.res["R2_train"] = r2_score(self.y_train, self.train_predict)
        self.res["Corr_train"] = pearsonr(self.y_train, self.train_predict)[0]
        self.res["RMSE_train"] = sqrt(mean_squared_error(self.y_train, self.train_predict))
        self.res["R2_test"] = r2_score(self.y_test, self.test_predict)
        self.res["Corr_test"] = pearsonr(self.y_test, self.test_predict)[0]
        self.res["RMSE_test"] = sqrt(mean_squared_error(self.y_test, self.test_predict))

        print("=========SUMMARY===============================================================")
        print("Mean of R2 from test sets: ")
        print(sum(statistics["r2"]) / len(statistics["r2"]))
        print("Mean of Corr from test sets: ")
        print(sum(statistics["corr"]) / len(statistics["corr"]))
        print("Mean of RMSE from test sets: ")
        print(sum(statistics["rmse"]) / len(statistics["rmse"]))

        return self.res

    def out2CSV(self):
        dimension_lis = []
        dimension_lis_index = []
        for i, d in enumerate(["time", "lat", "lon"]):
            if d not in self.feature_names:
                dimension_lis.append(d)
                dimension_lis_index.append(i)
        dimension_lis_index = tuple(dimension_lis_index)

        with open(os.path.join(self.out_path, self.f_name + '_prediction_test.csv'), "w") as csvfile:
            c = csv.writer(csvfile, delimiter=',')
            csv_header = [self.y_name, self.y_name + '_pred'] + self.feature_names + dimension_lis
            c.writerow(csv_header)
            var_matrix = np.c_[self.y_test, self.test_predict, self.X_test, self.X_dim_test[:, dimension_lis_index]]
            for row in var_matrix:
                c.writerow(row)

        with open(os.path.join(self.out_path, self.f_name + '_prediction_train.csv'), "w") as csvfile:
            c = csv.writer(csvfile, delimiter=',')
            csv_header = [self.y_name, self.y_name + '_pred'] + self.feature_names + dimension_lis
            c.writerow(csv_header)
            var_matrix = np.c_[self.y_train, self.train_predict, self.X_train, self.X_dim_train[:, dimension_lis_index]]
            for row in var_matrix:
                c.writerow(row)

        with open(os.path.join(self.out_path, self.f_name + '_prediction.csv'), "w") as csvfile:
            c = csv.writer(csvfile, delimiter=',')
            csv_header = [self.y_name, self.y_name + '_pred'] + self.feature_names + dimension_lis
            c.writerow(csv_header)
            var_matrix = np.c_[self.y_test, self.test_predict, self.X_test, self.X_dim_test[:, dimension_lis_index]]
            for row in var_matrix:
                c.writerow(row)
            var_matrix = np.c_[self.y_train, self.train_predict, self.X_train, self.X_dim_train[:, dimension_lis_index]]
            for row in var_matrix:
                c.writerow(row)

    def out2NC(self):
        y_name = self.y_name
        key_x_name = self.key_x_name

        fh_ref_train = Dataset(os.path.join(self.in_path, self.f_name + '_train.nc'), 'r')
        fh_ref_test = Dataset(os.path.join(self.in_path, self.f_name + '_test.nc'), 'r')
        fh_dic = {}
        fh_dic["predict"] = Dataset(os.path.join(self.out_path, self.f_name + '_prediction.nc'), 'w')
        fh_dic["train"] = Dataset(os.path.join(self.out_path, self.f_name + '_prediction_train.nc'), 'w')
        fh_dic["test"] = Dataset(os.path.join(self.out_path, self.f_name + '_prediction_test.nc'), "w")

        for name, dim in fh_ref_train.dimensions.items():
            for fh in fh_dic.values():
                fh.createDimension(name, len(dim))

        for v_name, varin in fh_ref_train.variables.items():
            if v_name in ['time', 'lat', 'lon']:
                for fh in fh_dic.values():
                    outVar = fh.createVariable(v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    outVar[:] = varin[:]
            if v_name == y_name:
                for fh in fh_dic.values():
                    outVar = fh.createVariable(v_name + '_predicted', varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    outVar = fh.createVariable(v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    y_train = fh_ref_train.variables[y_name][:]
                    y_test = fh_ref_test.variables[y_name][:]
                    outVar[:] = ma.array(y_train.filled(0) + y_test.filled(0),
                                         mask=ma.array([*map(and_, y_train.mask, y_test.mask)]))
            if v_name == key_x_name:
                for fh in fh_dic.values():
                    outVar = fh.createVariable(v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    key_x_train = fh_ref_train.variables[key_x_name][:]
                    key_x_test = fh_ref_test.variables[key_x_name][:]
                    outVar[:] = ma.array(key_x_train.filled(0) + key_x_test.filled(0),
                                         mask=ma.array([*map(and_, key_x_train.mask, key_x_test.mask)]))

        time = fh_ref_train.variables['time'][:]
        lat = fh_ref_train.variables['lat'][:]
        lon = fh_ref_train.variables['lon'][:]

        for idx, smap_value in enumerate(self.train_predict):
            time_index = find_index(time, self.X_dim_train[idx, 0])
            lat_index = find_index(lat, self.X_dim_train[idx, 1])
            lon_index = find_index(lon, self.X_dim_train[idx, 2])
            fh_dic["predict"].variables[y_name + '_predicted'][time_index, lat_index, lon_index] = smap_value
            fh_dic["train"].variables[y_name + '_predicted'][time_index, lat_index, lon_index] = smap_value
        for idx, smap_value in enumerate(self.test_predict):
            time_index = find_index(time, self.X_dim_test[idx, 0])
            lat_index = find_index(lat, self.X_dim_test[idx, 1])
            lon_index = find_index(lon, self.X_dim_test[idx, 2])
            fh_dic["predict"].variables[y_name + '_predicted'][time_index, lat_index, lon_index] = smap_value
            fh_dic["test"].variables[y_name + '_predicted'][time_index, lat_index, lon_index] = smap_value

        for fh in fh_dic.values():
            ma_predic = ma.getmaskarray(fh.variables[y_name + '_predicted'][:])
            fh.variables[y_name][:] = ma.array(fh.variables[y_name][:], mask=ma_predic)
            fh.variables[key_x_name][:] = ma.array(fh.variables[key_x_name][:], mask=ma_predic)

        fh_ref_train.close()
        fh_ref_test.close()
        fh_dic["predict"].close()
        fh_dic["train"].close()
        fh_dic["test"].close()

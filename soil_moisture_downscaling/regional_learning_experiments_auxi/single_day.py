# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from soil_moisture_downscaling.utils import get_out_path
from soil_moisture_downscaling.machine_learning import ApplyML_RegionalLearning
from soil_moisture_downscaling.plot import results_plot_regional_learning
from soil_moisture_downscaling.converter import convert2csv
from soil_moisture_downscaling.utils import concat_csv_files
from soil_moisture_downscaling.regional_learning_experiments_auxi import merge_prediction

import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime


def apply_ML_models(big_folder, experiment_folder, learning_type, chosen_dic, test_results, query_path, case_folder,
                    model, predict_diff, feature_folder, cv_type, search_type, scorer, query=None, query_index=None,
                    real_gap=False):
    for txt_file in os.listdir(query_path):
        if txt_file.startswith("queries") and txt_file.endswith(".txt"):
            if query is None or txt_file[:-4] == query:
                with open(os.path.join(query_path, txt_file)) as f:
                    lines = f.read().splitlines()
                    y_name = lines[0]
                    for area_folder in chosen_dic:
                        for doy_folder in os.listdir(os.path.join(big_folder, experiment_folder, "Experiment_Data", case_folder,
                                                                  area_folder)):
                            if doy_folder.startswith("20"):
                                for i in range(1, len(lines)):
                                    index = lines[i].split(" ")[0]
                                    if query_index is None or index == str(query_index):
                                        print(query, query_index)
                                        key_x_name = lines[i][2:].split(",")[0].strip()
                                        selected_features = [x.strip() for x in lines[i][2:].split(",")]

                                        case_path = os.path.join(big_folder,
                                                                 experiment_folder,
                                                                 "Experiment_Data",
                                                                 case_folder,
                                                                 area_folder,
                                                                 doy_folder)
                                        in_path = os.path.join(case_path, *feature_folder)
                                        out_path = get_out_path(os.path.join(big_folder,
                                                                             experiment_folder, case_folder, area_folder,
                                                                             doy_folder,
                                                                             "_".join([y_name, model]), index))
                                        # seeds = [192, 5029, 9280, 7793,  669, 2210, 5293, 7097, 1488, 1037]
                                        seeds = [192, 5029, 9280, 7793, 669]
                                        # seeds = [192]

                                        csvfile = open(os.path.join(out_path, 'results.csv'), "w")
                                        c = csv.writer(csvfile, delimiter=',')
                                        csv_header = ["Number_train", "Number_test",
                                                      "R2_train_before", "Corr_train_before",
                                                      "RMSE_train_before", "ubRMSE_train_before",
                                                      "Bias_train_before",
                                                      "R2_test_before", "Corr_test_before",
                                                      "RMSE_test_before", "ubRMSE_test_before",
                                                      "Bias_test_before",
                                                      "R2_train", "Corr_train", "RMSE_train", "ubRMSE_train", "Bias_train",
                                                      "R2_test", "Corr_test", "RMSE_test", "ubRMSE_test", "Bias_test"]
                                        c.writerow(csv_header)

                                        ml = ApplyML_RegionalLearning(learning_type=learning_type,
                                                                      in_path=in_path,
                                                                      train_file=learning_type,
                                                                      test_file="test",
                                                                      y_name=y_name,
                                                                      key_x_name=key_x_name,
                                                                      model=model,
                                                                      out_path=out_path,
                                                                      seeds=seeds,
                                                                      predict_diff=predict_diff,
                                                                      selected_features=selected_features,
                                                                      test_results=test_results,
                                                                      verbose=True,
                                                                      log=True,
                                                                      real_gap=real_gap)
                                        res = ml.apply(cv_type=cv_type,
                                                       search_type=search_type,
                                                       scorer=scorer)
                                        c.writerow([res[key] for key in csv_header])
                                        ml.out2NC()
                                        ml.out2CSV()
                                        ml.clean_up()
                                        print("============================================================")

                                        csvfile.close()
                                        results_plot_regional_learning(out_path)


def generate_merged_predictions(big_folder, origi_folder, experiment_folder, learning_type, chosen_dic, tuning_case, out_folder, var_names, model,
                                feature_index, base_folder, real_gap):
    for area_folder in chosen_dic:
        for doy_folder in os.listdir(os.path.join(big_folder, experiment_folder, "Experiment_Data", tuning_case,
                                     area_folder)):
            if doy_folder.startswith("20"):
                base_path = os.path.join(big_folder, experiment_folder, "Experiment_Data", tuning_case,
                                         area_folder, doy_folder, *base_folder)
                out_path = get_out_path(os.path.join(big_folder, experiment_folder, "Experiment_Data",
                                                     tuning_case, area_folder, doy_folder, model, out_folder))
                merge_prediction(big_folder, experiment_folder, base_path, out_path, var_names,
                                 [tuning_case, area_folder, doy_folder, model],
                                 feature_index,
                                 fname="test.nc",
                                 predicted_name="test_predicted.nc")
                convert2csv(out_path, f_name="test", doy=doy_folder)
                if "spatial" in learning_type:
                    merge_prediction(big_folder, experiment_folder, base_path, out_path, var_names,
                                     [tuning_case, area_folder, doy_folder, model],
                                     feature_index,
                                     fname="train_spatial.nc",
                                     predicted_name="train_spatial_predicted.nc")
                    convert2csv(out_path, f_name="train_spatial", doy=doy_folder)
                if "temporal" in learning_type:
                    if real_gap:
                        for doy_nc in os.listdir(os.path.join(base_path, "train_temporal")):
                            if doy_nc.endswith(".nc"):
                                merge_prediction(big_folder, experiment_folder, os.path.join(base_path, "train_temporal"),
                                                 os.path.join(out_path, "train_temporal"), var_names,
                                                 [tuning_case, area_folder, doy_folder, model],
                                                 feature_index,
                                                 fname=doy_nc,
                                                 predicted_name=doy_nc[:-3] + "_predicted.nc",
                                                 real_gap=True)
                                convert2csv(os.path.join(out_path, "train_temporal"), f_name=doy_nc[:-3],
                                            doy=doy_nc[:-3].split("_")[-1])
                        concat_csv_files(in_path=os.path.join(out_path, "train_temporal"),
                                         out_path=out_path,
                                         out_file="train_temporal.csv")
                    else:
                        merge_prediction(big_folder, experiment_folder, base_path, out_path, var_names,
                                         [tuning_case, area_folder, doy_folder, model],
                                         feature_index,
                                         fname="train_temporal.nc",
                                         predicted_name="train_temporal_predicted.nc")
                        temporal_df = pd.read_csv(os.path.join(big_folder, origi_folder, area_folder, doy_folder, "train_temporal.csv"))
                        temporal_doy = np.array(temporal_df[["doy"]])[0][0]
                        convert2csv(out_path, f_name="train_temporal", doy=datetime.strptime(str(temporal_doy), '%Y%j').strftime('%Y%m%d'))
                    if "spatial" in learning_type:
                        concat_csv_files(in_path=out_path,
                                         out_path=out_path,
                                         out_file="train_spatial_temporal.csv",
                                         in_file_lis=["train_spatial.csv", "train_temporal.csv"])

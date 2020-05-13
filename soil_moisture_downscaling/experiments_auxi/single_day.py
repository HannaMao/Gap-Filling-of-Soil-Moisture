# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..train_test_split import shrink_expand_split
from ..machine_learning import ApplyML
from ..plot import results_plot
from ..converter import convert2csv, convert2csv_strip_scanner
from ..feature_engineering import merge_prediction
from ..utils import get_out_path
from . import area_dic_spatial
from ..feature_engineering import historical_values_search
from soil_moisture_downscaling.plot import plot_single_variable

import os
import csv
import numpy as np


def vertical_split(case_folder, split_type, n_cells):
    for area_folder in area_dic_spatial:
        for doy_folder in os.listdir(os.path.join("STTL_Spatial_Comparison_Experiments", "Original_Data", area_folder)):
            if doy_folder.startswith("20"):
                in_path = os.path.join("STTL_Spatial_Comparison_Experiments", "Original_Data", area_folder, doy_folder)
                out_path = get_out_path(os.path.join("STTL_Spatial_Comparison_Experiments",
                                                     "Experiment_Data",
                                                     case_folder,
                                                     area_folder,
                                                     doy_folder,
                                                     split_type,
                                                     "base"))
                print(in_path)
                for nc_file in os.listdir(in_path):
                    if nc_file.endswith(".nc"):
                        for n_cell in n_cells:
                            shrink_expand_split(in_path=in_path,
                                                f_name=nc_file[:-3],
                                                n_cells=n_cell,
                                                out_path=out_path,
                                                split_type=split_type)
                        convert2csv(out_path)

                for nc_file in os.listdir(out_path):
                    if nc_file.endswith(".nc"):
                        plot_single_variable(out_path, [nc_file[:-3]], "soil_moisture", out_path, "usa",
                                             "cm**3/cm**3",
                                             v_min=0.0, v_max=0.5)


def apply_ML_models(query_path, case_folder, model, predict_diff, split_type, feature_folder, n_cells, resolution,
                    cv_type, search_type, scorer, query=None, query_index=None):
    for txt_file in os.listdir(query_path):
        if txt_file.startswith("queries") and txt_file.endswith(".txt"):
            if query is None or txt_file[:-4] == query:
                print(query, end=" ")
                with open(os.path.join(query_path, txt_file)) as f:
                    lines = f.read().splitlines()
                    y_name = lines[0]
                    for area_folder in area_dic_spatial:
                        for doy_folder in os.listdir(os.path.join("STTL_Spatial_Comparison_Experiments",
                                                                  "Experiment_Data",
                                                                  case_folder,
                                                                  area_folder)):
                            if doy_folder.startswith("20"):
                                for i in range(1, len(lines)):
                                    index = lines[i].split(" ")[0]
                                    if query_index is None or index == str(query_index):
                                        print(query_index)
                                        key_x_name = lines[i][2:].split(",")[0].strip()
                                        selected_features = [x.strip() for x in lines[i][2:].split(",")]

                                        case_path = os.path.join("STTL_Spatial_Comparison_Experiments",
                                                                 "Experiment_Data",
                                                                 case_folder,
                                                                 area_folder,
                                                                 doy_folder)
                                        in_path = os.path.join(case_path, split_type, *feature_folder)
                                        out_path = get_out_path(os.path.join("STTL_Spatial_Comparison_Experiments", case_folder, area_folder,
                                                                             doy_folder,
                                                                             "_".join([y_name, model]), index))

                                        key_x_name = key_x_name
                                        # seeds = [192, 5029, 9280, 7793, 669, 2210, 5293, 7097, 1488, 1037]
                                        seeds = [192, 5029, 9280, 7793, 669]
                                        # seeds = [192]

                                        csvfile = open(os.path.join(out_path, 'results.csv'), "w")
                                        c = csv.writer(csvfile, delimiter=',')
                                        csv_header = ["File",
                                                      "Number_train", "Number_test",
                                                      "R2_train_before", "Corr_train_before",
                                                      "RMSE_train_before", "ubRMSE_train_before",
                                                      "R2_test_before", "Corr_test_before",
                                                      "RMSE_test_before", "ubRMSE_test_before",
                                                      "R2_train", "Corr_train", "RMSE_train", "ubRMSE_train",
                                                      "R2_test", "Corr_test", "RMSE_test", "ubRMSE_test"]
                                        c.writerow(csv_header)

                                        for n_cell in n_cells:
                                            f_name = str(n_cell * resolution) + "km"
                                            ml = ApplyML(in_path=in_path,
                                                         f_name=f_name,
                                                         y_name=y_name,
                                                         key_x_name=key_x_name,
                                                         model=model,
                                                         out_path=out_path,
                                                         seeds=seeds,
                                                         predict_diff=predict_diff,
                                                         selected_features=selected_features,
                                                         verbose=True,
                                                         log=True)
                                            res = ml.apply(cv_type=cv_type,
                                                           search_type=search_type,
                                                           scorer=scorer)
                                            c.writerow([res[key] for key in csv_header])
                                            ml.out2NC()
                                            ml.out2CSV()
                                            ml.clean_up()
                                            print("============================================================")

                                        csvfile.close()
                                        results_plot(out_path)


def generate_merged_predictions(tuning_case, out_folder, var_names, model, feature_index, base_folder,
                                split_type, n_cells):
    for area_folder in area_dic_spatial:
        for doy_folder in os.listdir(os.path.join("STTL_Spatial_Comparison_Experiments", "Experiment_Data", tuning_case, area_folder)):
            if doy_folder.startswith("20"):
                base_path = os.path.join("STTL_Spatial_Comparison_Experiments", "Experiment_Data", tuning_case, area_folder, doy_folder,
                                         split_type, *base_folder)
                out_path = get_out_path(os.path.join("STTL_Spatial_Comparison_Experiments", "Experiment_Data", tuning_case, area_folder, doy_folder,
                                                     split_type, model, out_folder))

                merge_prediction(base_path, out_path, var_names,
                                 [tuning_case, area_folder, doy_folder, model],
                                 feature_index,
                                 n_cells)
                if tuning_case.split("_")[0] != "adaptive":
                    convert2csv(out_path)
                else:
                    convert2csv_strip_scanner(out_path)


# def generate_hist_as_extra_feature():
#     hist_var_list = ["sigma0_vh_aggregated", "soil_moisture"]
#     hist_n_threshold = 50
#     analyze_hist_r2(hist_var_list, hist_n_threshold, np.arange(0.5, 0.8, 0.05))
#
#     hist_r2_thres = 0.6
#     for data_folder in os.listdir(experiment_path):
#         if os.path.isdir(os.path.join(experiment_path, data_folder)):
#             area = data_folder.split("_")[0]
#             data_path = os.path.join(experiment_path, data_folder)
#             for n in range(1, n_strips + 1):
#                 historical_values_search(data_path=os.path.join(data_path, split_type),
#                                          f_name=str(n * multiplier * resolution) + "km",
#                                          search_path=os.path.join("Data", "Sentinel", "usa"),
#                                          lat1=area_dic[area]["lat1"],
#                                          lat2=area_dic[area]["lat2"],
#                                          lon1=area_dic[area]["lon1"],
#                                          lon2=area_dic[area]["lon2"],
#                                          doy=area_dic[area]["doy"],
#                                          var_list=hist_var_list,
#                                          out_path=get_out_path(
#                                              os.path.join(data_path, "extra_features")),
#                                          output=True,
#                                          verbose=True,
#                                          n_threshold=hist_n_threshold,
#                                          r2_threshold=hist_r2_thres)



# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from soil_moisture_downscaling.utils import timeit
from soil_moisture_downscaling.experiments_auxi.single_day import *

import os


def single_day_experiment(cv_type, search_type, scorer, model, split=True):
    timeit()

    split_type = "vertical_split_single"
    queries_path = os.path.join("STTL_Spatial_Comparison_Experiments", "queries_single_day")
    case_folder = "_".join(map(str, [cv_type, search_type, scorer]))

    n_cells = [18, 30, 44, 60]  #1:4, 1:2, 1:1, 2:1
    resolution = 3

    if split:
        vertical_split(case_folder, split_type, n_cells)

    queries_dic = {"queries_tb_v_disaggregated": [(["base"], [1])]}

    for query in queries_dic:
        for feature_folder, f_indices in queries_dic[query]:
            for feature_index in f_indices:
                predict_diff = True
                apply_ML_models(query_path=queries_path,
                                case_folder=case_folder,
                                model=model,
                                predict_diff=predict_diff,
                                split_type=split_type,
                                feature_folder=feature_folder,
                                n_cells=n_cells,
                                resolution=resolution,
                                cv_type=cv_type,
                                search_type=search_type,
                                scorer=scorer,
                                query=query,
                                query_index=feature_index)

    generate_merged_predictions(tuning_case=case_folder,
                                out_folder="bt_1",
                                var_names=["tb_v_disaggregated"],
                                model=model,
                                feature_index="1",
                                base_folder=["base"],
                                split_type=split_type,
                                n_cells=n_cells)

    generate_merged_predictions(tuning_case=case_folder,
                                out_folder="bt_2",
                                var_names=["tb_v_disaggregated"],
                                model=model,
                                feature_index="2",
                                base_folder=["base"],
                                split_type=split_type,
                                n_cells=n_cells)

    generate_merged_predictions(tuning_case=case_folder,
                                out_folder="bt_3",
                                var_names=["tb_v_disaggregated"],
                                model=model,
                                feature_index="3",
                                base_folder=["base"],
                                split_type=split_type,
                                n_cells=n_cells)

    generate_merged_predictions(tuning_case=case_folder,
                                out_folder="bt_4",
                                var_names=["tb_v_disaggregated"],
                                model=model,
                                feature_index="4",
                                base_folder=["base"],
                                split_type=split_type,
                                n_cells=n_cells)

    queries_dic = {"queries_soil_moisture": [([model, "bt_1"], [1, 2])]}

    for query in queries_dic:
        for feature_folder, f_indices in queries_dic[query]:
            for feature_index in f_indices:
                predict_diff = False
                apply_ML_models(queries_path,
                                case_folder,
                                model,
                                predict_diff,
                                split_type,
                                feature_folder,
                                n_cells,
                                resolution,
                                cv_type,
                                search_type,
                                scorer,
                                query,
                                feature_index)

    queries_dic = {"queries_soil_moisture": [([model, "bt_2"], [4])]}

    for query in queries_dic:
        for feature_folder, f_indices in queries_dic[query]:
            for feature_index in f_indices:
                predict_diff = False
                apply_ML_models(queries_path,
                                case_folder,
                                model,
                                predict_diff,
                                split_type,
                                feature_folder,
                                n_cells,
                                resolution,
                                cv_type,
                                search_type,
                                scorer,
                                query,
                                feature_index)

    queries_dic = {"queries_soil_moisture": [([model, "bt_3"], [5])]}

    for query in queries_dic:
        for feature_folder, f_indices in queries_dic[query]:
            for feature_index in f_indices:
                predict_diff = False
                apply_ML_models(queries_path,
                                case_folder,
                                model,
                                predict_diff,
                                split_type,
                                feature_folder,
                                n_cells,
                                resolution,
                                cv_type,
                                search_type,
                                scorer,
                                query,
                                feature_index)

    queries_dic = {"queries_soil_moisture": [([model, "bt_4"], [6])]}

    for query in queries_dic:
        for feature_folder, f_indices in queries_dic[query]:
            for feature_index in f_indices:
                predict_diff = False
                apply_ML_models(queries_path,
                                case_folder,
                                model,
                                predict_diff,
                                split_type,
                                feature_folder,
                                n_cells,
                                resolution,
                                cv_type,
                                search_type,
                                scorer,
                                query,
                                feature_index)


if __name__ == "__main__":
    single_day_experiment(cv_type=None, search_type=None, scorer="corr", model="RandomForestModel", split=True)
    single_day_experiment(cv_type="shuffle", search_type="random", scorer="corr", model="XGBoostModel", split=False)
    single_day_experiment(cv_type="adaptive", search_type="random", scorer="corr", model="XGBoostModel", split=False)















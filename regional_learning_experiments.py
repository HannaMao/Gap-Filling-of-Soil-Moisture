# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from soil_moisture_downscaling.utils import timeit
from soil_moisture_downscaling.regional_learning_experiments_auxi.single_day import *
from shutil import copyfile
from soil_moisture_downscaling.regional_learning_experiments_auxi import area_dic


def copy_to_base(big_folder, origi_data_folder, experiment_folder, learning_type, chosen_dic, case_folder, real_gap=False):

    for area in chosen_dic:
        for doy in os.listdir(os.path.join(big_folder, origi_data_folder, area)):
            if doy.startswith("20"):
                out_folder = get_out_path(os.path.join(big_folder, experiment_folder,
                                                       "Experiment_Data", case_folder, area, doy, "base"))
                in_folder = os.path.join(big_folder, origi_data_folder, area, doy)
                # copy test.csv
                copyfile(os.path.join(in_folder, "test.csv"),
                         os.path.join(out_folder, "test.csv"))
                # copy [learning_type].csv
                copyfile(os.path.join(in_folder, learning_type + ".csv"),
                         os.path.join(out_folder, learning_type + ".csv"))
                # copy test.nc
                copyfile(os.path.join(in_folder, "test.nc"),
                         os.path.join(out_folder, "test.nc"))
                if "spatial" in learning_type:
                    copyfile(os.path.join(in_folder, "train_spatial.nc"),
                             os.path.join(out_folder, "train_spatial.nc"))
                if "temporal" in learning_type:
                    if real_gap:
                        get_out_path(os.path.join(out_folder, "train_temporal"))
                        for nc_file in os.listdir(os.path.join(in_folder, "train_temporal")):
                            if nc_file.endswith(".nc"):
                                copyfile(os.path.join(in_folder, "train_temporal", nc_file),
                                         os.path.join(out_folder, "train_temporal", nc_file))
                    else:
                        for nc_file in os.listdir(in_folder):
                            if nc_file.endswith(".nc"):
                                copyfile(os.path.join(in_folder, nc_file),
                                         os.path.join(out_folder, nc_file))


def regional_learning_experiment(big_folder, origi_data_folder, experiment_folder, learning_type, chosen_dic,
                                 test_results, cv_type, search_type, scorer, model, real_gap=False):
    timeit()

    queries_path = os.path.join(big_folder, "queries_single_day")
    case_folder = "_".join(map(str, [cv_type, search_type, scorer]))

    copy_to_base(big_folder, origi_data_folder, experiment_folder, learning_type, chosen_dic, case_folder,
                 real_gap=real_gap)

    queries_dic = {"queries_tb_v_disaggregated": [(["base"], [1, 2, 3, 4])]}

    for query in queries_dic:
        for feature_folder, f_indices in queries_dic[query]:
            for feature_index in f_indices:
                predict_diff = True
                apply_ML_models(big_folder,
                                experiment_folder,
                                learning_type,
                                chosen_dic,
                                test_results,
                                queries_path,
                                case_folder,
                                model,
                                predict_diff,
                                feature_folder,
                                cv_type,
                                search_type,
                                scorer,
                                query,
                                feature_index,
                                real_gap)

    generate_merged_predictions(big_folder,
                                origi_data_folder,
                                experiment_folder,
                                learning_type,
                                chosen_dic,
                                tuning_case=case_folder,
                                out_folder="bt_1",
                                var_names=["tb_v_disaggregated"],
                                model=model,
                                feature_index="1",
                                base_folder=["base"],
                                real_gap=real_gap)

    generate_merged_predictions(big_folder,
                                origi_data_folder,
                                experiment_folder,
                                learning_type,
                                chosen_dic,
                                tuning_case=case_folder,
                                out_folder="bt_2",
                                var_names=["tb_v_disaggregated"],
                                model=model,
                                feature_index="2",
                                base_folder=["base"],
                                real_gap=real_gap)

    generate_merged_predictions(big_folder,
                                origi_data_folder,
                                experiment_folder,
                                learning_type,
                                chosen_dic,
                                tuning_case=case_folder,
                                out_folder="bt_3",
                                var_names=["tb_v_disaggregated"],
                                model=model,
                                feature_index="3",
                                base_folder=["base"],
                                real_gap=real_gap)

    generate_merged_predictions(big_folder,
                                origi_data_folder,
                                experiment_folder,
                                learning_type,
                                chosen_dic,
                                tuning_case=case_folder,
                                out_folder="bt_4",
                                var_names=["tb_v_disaggregated"],
                                model=model,
                                feature_index="4",
                                base_folder=["base"],
                                real_gap=real_gap)

    queries_dic = {"queries_soil_moisture": [([model, "bt_1"], [1, 2, 3, 7])]}
    queries_dic = {"queries_soil_moisture": [([model, "bt_1"], [8])]}

    for query in queries_dic:
        for feature_folder, f_indices in queries_dic[query]:
            for feature_index in f_indices:
                predict_diff = False
                apply_ML_models(big_folder,
                                experiment_folder,
                                learning_type,
                                chosen_dic,
                                test_results,
                                queries_path,
                                case_folder,
                                model,
                                predict_diff,
                                feature_folder,
                                cv_type,
                                search_type,
                                scorer,
                                query,
                                feature_index,
                                real_gap)

    queries_dic = {"queries_soil_moisture": [([model, "bt_2"], [4])]}

    for query in queries_dic:
        for feature_folder, f_indices in queries_dic[query]:
            for feature_index in f_indices:
                predict_diff = False
                apply_ML_models(big_folder,
                                experiment_folder,
                                learning_type,
                                chosen_dic,
                                test_results,
                                queries_path,
                                case_folder,
                                model,
                                predict_diff,
                                feature_folder,
                                cv_type,
                                search_type,
                                scorer,
                                query,
                                feature_index,
                                real_gap)

    queries_dic = {"queries_soil_moisture": [([model, "bt_3"], [5])]}

    for query in queries_dic:
        for feature_folder, f_indices in queries_dic[query]:
            for feature_index in f_indices:
                predict_diff = False
                apply_ML_models(big_folder,
                                experiment_folder,
                                learning_type,
                                chosen_dic,
                                test_results,
                                queries_path,
                                case_folder,
                                model,
                                predict_diff,
                                feature_folder,
                                cv_type,
                                search_type,
                                scorer,
                                query,
                                feature_index,
                                real_gap)

    queries_dic = {"queries_soil_moisture": [([model, "bt_4"], [6])]}

    for query in queries_dic:
        for feature_folder, f_indices in queries_dic[query]:
            for feature_index in f_indices:
                predict_diff = False
                apply_ML_models(big_folder,
                                experiment_folder,
                                learning_type,
                                chosen_dic,
                                test_results,
                                queries_path,
                                case_folder,
                                model,
                                predict_diff,
                                feature_folder,
                                cv_type,
                                search_type,
                                scorer,
                                query,
                                feature_index,
                                real_gap)


if __name__ == "__main__":
    for model in ["RandomForestModel"]:
        # regional learning experiemnts: spatial, temporal, spatial & temporal
        regional_learning_experiment(big_folder="regional_learning_Experiments",
                                     origi_data_folder="Original_Data",
                                     experiment_folder="Spatial_Experiments", learning_type="train_spatial",
                                     chosen_dic=area_dic,
                                     test_results=True, cv_type=None, search_type=None, scorer="corr", model=model)
        regional_learning_experiment(big_folder="regional_learning_Experiments",
                                     origi_data_folder="Original_Data",
                                     experiment_folder="Temporal_Experiments", learning_type="train_temporal",
                                     chosen_dic=area_dic,
                                     test_results=True, cv_type=None, search_type=None, scorer="corr", model=model)
        regional_learning_experiment(big_folder="regional_learning_Experiments",
                                     origi_data_folder="Original_Data",
                                     experiment_folder="Spatial_Temporal_Experiments",
                                     learning_type="train_spatial_temporal", chosen_dic=area_dic,
                                     test_results=True, cv_type=None, search_type=None, scorer="corr", model=model)

        # temporal limitation exploration
        regional_learning_experiment(big_folder="STTL_Temporal_Comparison_Experiments",
                                     origi_data_folder="sttl_experiment_most_recent",
                                     experiment_folder="Most_Recent", learning_type="train_temporal",
                                     chosen_dic=area_dic,
                                     test_results=True, cv_type=None, search_type=None, scorer="corr",
                                     model=model)
        regional_learning_experiment(big_folder="STTL_Temporal_Comparison_Experiments",
                                     origi_data_folder="sttl_experiment_cohort1",
                                     experiment_folder="Cohort_1", learning_type="train_temporal",
                                     chosen_dic=area_dic,
                                     test_results=True, cv_type=None, search_type=None, scorer="corr",
                                     model=model)
        regional_learning_experiment(big_folder="STTL_Temporal_Comparison_Experiments",
                                     origi_data_folder="sttl_experiment_cohort2",
                                     experiment_folder="Cohort_2", learning_type="train_temporal",
                                     chosen_dic=area_dic,
                                     test_results=True, cv_type=None, search_type=None, scorer="corr",
                                     model=model)
        regional_learning_experiment(big_folder="STTL_Temporal_Comparison_Experiments",
                                     origi_data_folder="sttl_experiment_cohort3",
                                     experiment_folder="Cohort_3", learning_type="train_temporal",
                                     chosen_dic=area_dic,
                                     test_results=True, cv_type=None, search_type=None, scorer="corr",
                                     model=model)

        # filling real gaps, arizona, south dakota, arkansas
        regional_learning_experiment(big_folder="regional_learning_Real_Gaps",
                                     origi_data_folder="Original_Data",
                                     experiment_folder="Spatial_Experiments",
                                     learning_type="train_spatial",
                                     chosen_dic=area_dic,
                                     test_results=False, cv_type=None, search_type=None, scorer="corr", model=model,
                                     real_gap=True)
        regional_learning_experiment(big_folder="regional_learning_Real_Gaps",
                                     origi_data_folder="Original_Data",
                                     experiment_folder="Temporal_Experiments",
                                     learning_type="train_temporal",
                                     chosen_dic=area_dic,
                                     test_results=False, cv_type=None, search_type=None, scorer="corr", model=model,
                                     real_gap=True)
        regional_learning_experiment(big_folder="regional_learning_Real_Gaps",
                                     origi_data_folder="Original_Data",
                                     experiment_folder="Spatial_Temporal_Experiments",
                                     learning_type="train_spatial_temporal",
                                     chosen_dic=area_dic,
                                     test_results=False, cv_type=None, search_type=None, scorer="corr", model=model,
                                     real_gap=True)

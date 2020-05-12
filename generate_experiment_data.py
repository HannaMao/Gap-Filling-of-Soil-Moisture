# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from data_preprocessing.preprocess import *
from data_preprocessing.rescaling import *
from data_preprocessing.utils import *
from data_preprocessing import states_index_dic
from data_preprocessing.plot import plot_single_variable
from data_preprocessing import area_filter, real_gap_area_filter
from data_preprocessing import area_dic
from data_preprocessing.feature_engineering import *
from data_preprocessing.merge import merge_various_variables_v2
from data_preprocessing.spatial_temporal_regional_learning import merge_various_variables_mask_v2
from data_preprocessing import convert2csv
from data_preprocessing.merge import merge_csv_files

import os


def check_area_availability():
    area_file_lists = {"arizona": {"spatial": [], "temporal": []},
                       "oklahoma": {"spatial": [], "temporal": []},
                       "iowa": {"spatial": [], "temporal": []}}
    arkansas_file_lists = {"arkansas": {"spatial": [], "temporal": []},}
    # print("Combine tiles of smap/sentinel data. Unit of radar backscatter is converted to db.")
    for doy in generate_doy("20150401", "20181001", "."):
        area_file = combine_to_3km_local(doy, 50, 24, -125, -66, "usa_filtered", convert_to_db=False, with_filter=True,
                                         area_filter=area_filter)
        if len(area_file) > 0:
            for area in arkansas_file_lists:
                for t in ["spatial", "temporal"]:
                    if len(area_file[area][t]) == 1:
                        arkansas_file_lists[area][t].append(list(area_file[area][t])[0])

    # mask using state boundaries
    for area in ["arkansas"]:
        for doy in generate_doy("20150401", "20181001", ""):
            if os.path.isfile(os.path.join("Data", "Sentinel", "usa_filtered", doy + ".nc")):
                out_path = get_out_path(os.path.join("Data", "Sentinel", "usa_states", "mask_by_state", area))
                removed_doy = mask_by_state(states_index_dic[area],
                                            file_in=os.path.join("Data", "Sentinel", "usa_filtered", doy + ".nc"),
                                            file_out=os.path.join(out_path, doy + ".nc"))
                if removed_doy:
                    if removed_doy in arkansas_file_lists[area]["spatial"]:
                        arkansas_file_lists[area]["spatial"].remove(removed_doy)
                    if removed_doy in arkansas_file_lists[area]["temporal"]:
                        arkansas_file_lists[area]["temporal"].remove(removed_doy)

    save_pkl(arkansas_file_lists, "arkansas_file_lists.pkl")

    for area in ["arkansas"]:
        for f in os.listdir(os.path.join("Data", "Sentinel", "usa_states", "mask_by_state", area)):
            if f.endswith(".nc"):
                plot_single_variable(in_path=os.path.join("Data", "Sentinel", "usa_states", "mask_by_state", area),
                                     f_names=[f[:-3]],
                                     v_name="soil_moisture",
                                     out_path=os.path.join("Data", "Sentinel", "usa_states", "mask_by_state", area,
                                                           "plot"),
                                     type="usa",
                                     unit="cm**3/cm**3")


def check_area_availability_real_gaps():
    area_file_lists = {"arizona": {"spatial": [], "temporal1": [], "temporal2": []},
                       "oklahoma": {"spatial": [], "temporal1": [], "temporal2": []},
                       "iowa": {"spatial": [], "temporal": []}}
    arkansas_file_lists = {"arkansas": {"spatial": [], "temporal": []}}
    # print("Combine tiles of smap/sentinel data. Unit of radar backscatter is converted to db.")
    for doy in generate_doy("20150401", "20181001", "."):
        area_file = combine_to_3km_local(doy, 50, 24, -125, -66, "usa_filtered_real_gap", convert_to_db=False,
                                         with_filter=True, area_filter=real_gap_area_filter)
        if len(area_file) > 0:
            for area in arkansas_file_lists:
                for t in arkansas_file_lists[area]:
                    if len(area_file[area][t]) == 1:
                        arkansas_file_lists[area][t].append(list(area_file[area][t])[0])

    # mask using state boundaries
    for area in ["arkansas"]:
        for doy in generate_doy("20150401", "20181001", ""):
            if os.path.isfile(os.path.join("Data", "Sentinel", "usa_filtered_real_gap", doy + ".nc")):
                out_path = get_out_path(os.path.join("Data", "Sentinel", "usa_states_real_gap", "mask_by_state", area))
                removed_doy = mask_by_state(states_index_dic[area],
                                            file_in=os.path.join("Data", "Sentinel", "usa_filtered_real_gap", doy + ".nc"),
                                            file_out=os.path.join(out_path, doy + ".nc"))
                if removed_doy:
                    for t in arkansas_file_lists[area]:
                        if removed_doy in arkansas_file_lists[area][t]:
                            arkansas_file_lists[area][t].remove(removed_doy)
    save_pkl(arkansas_file_lists, "real_gap_arkansas_file_lists.pkl")

    for area in ["arkansas"]:
        for f in os.listdir(os.path.join("Data", "Sentinel", "usa_states_real_gap", "mask_by_state", area)):
            if f.endswith(".nc"):
                plot_single_variable(in_path=os.path.join("Data", "Sentinel", "usa_states_real_gap", "mask_by_state", area),
                                     f_names=[f[:-3]],
                                     v_name="soil_moisture",
                                     out_path=os.path.join("Data", "Sentinel", "usa_states_real_gap", "mask_by_state", area,
                                                           "plot"),
                                     type="usa",
                                     unit="cm**3/cm**3")


def expand_doys(doys, n_diff):
    expanded_doys = set()
    for doy in doys:
        expanded_doys.add(doy)
        for e_d in generate_most_recent_doys(doy, n_diff, ""):
            expanded_doys.add(e_d)
    return sorted(list(expanded_doys))


def basic_data_preprocessing():
    soil_fraction_upsample(50, 24, -125, -66, "3km", "usa")
    bulk_density_upsample(50, 24, -125, -66, "3km", "usa")
    landcover_upsample(50, 24, -125, -66, "3km", "usa")

    for doy in generate_doy("20150401", "20181001", "."):
        combine_to_3km_local(doy, 50, 24, -125, -66, "usa", convert_to_db=False)
    smap_sentinel_upscale("20150401", "20181001", ["sigma0_vv_aggregated", "sigma0_vh_aggregated"],
                          out_folder="usa_9km_all", output_all=True)
    smap_sentinel_upscale("20150401", "20181001", ["sigma0_vv_aggregated", "sigma0_vh_aggregated"],
                          out_folder="usa_9km_only", output_all=False)

    gpm_extract("GPM_3IMERGDF.05", "20150401", "20181001")
    all_doys = load_pkl("arkansas_all_doys.pkl")
    real_gap_all_doys = load_pkl("arkansas_real_gap_all_doys.pkl")
    expanded_doys = expand_doys(list(set(all_doys + real_gap_all_doys)), 3)
    todo_doys = []
    for doy in generate_doy("20150401", "20181001", ""):
        if doy in expanded_doys and doy + '.nc' not in os.listdir(os.path.join("Data", "GPM", "3km")):
            todo_doys.append(doy)
    print(todo_doys)
    print(len(todo_doys))
    gpm_downsample_nn_given_doys(todo_doys, 50, 24, -125, -66)

    for doy in generate_doy("20181002", "20181231", ""):
        smap_p_e_convert_to_nc_local(doy, 52, 22, -127, -64, "usa")
    smap_p_e_exact_downscale("20150401", "20181001")
    exact_downscale_resize_to_match_sentinel("20150401", "20181001")

    modis_lai_extract("20170201-20171014", "MCD15A3H.006_500m_aid0001")
    modis_lai_upsample("20170201", "20171014")
    modis_lai_fill_gap(os.path.join("Data", "MCD15A3H", "3km"), "20180902", "20181001")

    modis_lst_extract("20170201-20171014", "MOD11A1.006_1km_aid0001")
    modis_lst_upsample("20170201", "20171014")

    elevation_slope_extract()


def feature_engineering():
    all_doys = load_pkl("arkansas_all_doys.pkl")
    real_gap_all_doys = load_pkl("arkansas_real_gap_all_doys.pkl")
    expanded_doys = list(set(all_doys + real_gap_all_doys))
    for area in ["arkansas"]:
        for doy in expanded_doys:
            generate_nearly_covered_modis_lst(area,
                                              doy,
                                              area_dic[area]["lat1"],
                                              area_dic[area]["lat2"],
                                              area_dic[area]["lon1"],
                                              area_dic[area]["lon2"])

    todo_doys = []
    for doy in generate_doy("20150401", "20181001", ""):
        if doy in expanded_doys and doy + '.nc' not in os.listdir(
                os.path.join("Data", "gpm", "hist_added")):
            todo_doys.append(doy)
    print(todo_doys)
    print(len(todo_doys))
    generate_gpm_hist_by_doy(todo_doys)
    todo_doys = []
    for doy in generate_doy("20150401", "20181001", ""):
        if doy in expanded_doys and doy + '.nc' not in os.listdir(os.path.join("Data", "Sentinel", "usa_rb_hist_average_30")):
            todo_doys.append(doy)
    print(todo_doys)
    print(len(todo_doys))
    generate_rb_hist_average_time_window_by_doy(todo_doys, 30,
                                                ["sigma0_vh_aggregated", "sigma0_vv_aggregated",
                                                 "sigma0_vh_aggregated_9km_mean", "sigma0_vv_aggregated_9km_mean",
                                                 "beta_tbv_vv", "gamma_vv_xpol"])


if __name__ == "__main__":
    timeit()
    check_area_availability()
    basic_data_preprocessing()
    feature_engineering()

    area_pair_dic_list = {"most_recent": load_pkl("arkansas_most_recent_pairs.pkl"),
                          "cohort1": load_pkl("arkansas_cohort1_pairs.pkl"),
                          "cohort2": load_pkl("arkansas_cohort2_pairs.pkl"),
                          "cohort3": load_pkl("arkansas_cohort3_pairs.pkl")}

    for key in ["most_recent", "cohort1", "cohort2", "cohort3"]:
        for area in ["arkansas"]:
            for pair in area_pair_dic_list[key][area]:
                if pair[0] in os.listdir(os.path.join("sttl_experiment_" + key, area)):
                    v_max = 0.2 if area == "arizona" else 0.5
                    out_path = get_out_path(os.path.join("sttl_experiment_" + key, area, pair[0]))
                    test_mask = get_two_variables_mask(file_1=os.path.join("Data", "Sentinel", "usa_states",
                                                                           "mask_by_state", area, pair[0] + ".nc"),
                                                       file_2=os.path.join("Data", "Sentinel", "usa_states",
                                                                           "mask_by_state", area, pair[1] + ".nc"),
                                                       variable1="soil_moisture",
                                                       variable2="soil_moisture")

                    merge_various_variables_mask_v2(mask_array=test_mask,
                                                    mask_cut=True,
                                                    out_path=out_path,
                                                    out_file="test.nc",
                                                    lat1=area_dic[area]["lat1"],
                                                    lat2=area_dic[area]["lat2"],
                                                    lon1=area_dic[area]["lon1"],
                                                    lon2=area_dic[area]["lon2"],
                                                    area_name=area,
                                                    doy=pair[0],
                                                    n_hist=30,
                                                    selected_sentinel_fields=["soil_moisture", "tb_v_disaggregated"],
                                                    sentinel_path=os.path.join("Data", "Sentinel", "usa_states",
                                                                               "mask_by_state", area))
                    convert2csv(out_path, "test", doy=pair[0])
                    plot_single_variable(os.path.join("sttl_experiment_"+key, area, pair[0]),
                                         ["test"],
                                         "soil_moisture",
                                         os.path.join("sttl_experiment_" + key, area, pair[0]),
                                         "usa",
                                         "cm**3/cm**3",
                                         v_min=0.0,
                                         v_max=v_max)

                    merge_various_variables_mask_v2(mask_array=~test_mask,
                                                    mask_cut=True,
                                                    out_path=out_path,
                                                    out_file="train_spatial.nc",
                                                    lat1=area_dic[area]["lat1"],
                                                    lat2=area_dic[area]["lat2"],
                                                    lon1=area_dic[area]["lon1"],
                                                    lon2=area_dic[area]["lon2"],
                                                    area_name=area,
                                                    doy=pair[0],
                                                    n_hist=30,
                                                    selected_sentinel_fields=["soil_moisture", "tb_v_disaggregated"],
                                                    sentinel_path=os.path.join("Data", "Sentinel", "usa_states",
                                                                               "mask_by_state", area))
                    convert2csv(out_path, "train_spatial", doy=pair[0])
                    plot_single_variable(os.path.join("sttl_experiment_" + key, area, pair[0]),
                                         ["train_spatial"],
                                         "soil_moisture",
                                         os.path.join("sttl_experiment_" + key, area, pair[0]),
                                         "usa",
                                         "cm**3/cm**3",
                                         v_min=0.0,
                                         v_max=v_max)

                    merge_various_variables_mask_v2(mask_array=test_mask,
                                                    mask_cut=True,
                                                    out_path=out_path,
                                                    out_file="train_temporal.nc",
                                                    lat1=area_dic[area]["lat1"],
                                                    lat2=area_dic[area]["lat2"],
                                                    lon1=area_dic[area]["lon1"],
                                                    lon2=area_dic[area]["lon2"],
                                                    area_name=area,
                                                    doy=pair[1],
                                                    n_hist=30,
                                                    selected_sentinel_fields=["soil_moisture", "tb_v_disaggregated"],
                                                    sentinel_path=os.path.join("Data", "Sentinel", "usa_states",
                                                                               "mask_by_state", area))
                    convert2csv(out_path, "train_temporal", doy=pair[1])
                    plot_single_variable(os.path.join("sttl_experiment_" + key, area, pair[0]),
                                         ["train_temporal"],
                                         "soil_moisture",
                                         os.path.join("sttl_experiment_" + key, area, pair[0]),
                                         "usa",
                                         "cm**3/cm**3",
                                         v_min=0.0,
                                         v_max=v_max)
                    merge_csv_files(in_path=out_path,
                                    out_path=out_path,
                                    out_file="train_spatial_temporal.csv",
                                    in_file_lis=["train_spatial.csv", "train_temporal.csv"])

    for area in ["arkansas"]:
        for pair in area_pair_dic_list["most_recent"][area]:
            if pair[0] in os.listdir(os.path.join("spatial_experiment_data", area)):
                v_max = 0.2 if area == "arizona" else 0.5
                merge_various_variables_v2(sentinel_path=os.path.join("Data", "Sentinel", "usa_states", "mask_by_state",
                                                                      area),
                                           out_path=os.path.join("spatial_experiment_data", area, pair[0]),
                                           out_file="_".join([area, pair[0]]) + ".nc",
                                           lat1=area_dic[area]["lat1"],
                                           lat2=area_dic[area]["lat2"],
                                           lon1=area_dic[area]["lon1"],
                                           lon2=area_dic[area]["lon2"],
                                           area_name=area,
                                           doy=pair[0],
                                           n_hist=30,
                                           selected_sentinel_fields=["soil_moisture", "tb_v_disaggregated"])
                convert2csv(os.path.join("spatial_experiment_data", area, pair[0]), "_".join([area, pair[0]]))
                plot_single_variable(os.path.join("spatial_experiment_data", area, pair[0]),
                                     ["_".join([area, pair[0]])],
                                     "soil_moisture",
                                     os.path.join("spatial_experiment_data", area, pair[0]),
                                     "usa",
                                     "cm**3/cm**3",
                                     v_min=0.0,
                                     v_max=v_max)

    check_area_availability_real_gaps()
    real_gap_area_pair_dic = load_pkl("arkansas_real_gap_most_recent_pairs.pkl")
    for area in ["arkansas"]:
        for pair in real_gap_area_pair_dic[area]:
            if pair[0] in os.listdir(os.path.join("real_gap_sttl_experiment", area)):
                v_max = 0.2 if area == "arizona" else 0.5
                out_path = get_out_path(os.path.join("real_gap_sttl_experiment", area, pair[0]))

                merge_various_variables_v2(sentinel_path=os.path.join("Data", "Sentinel", "usa_states_real_gap",
                                                                      "mask_by_state", area),
                                           out_path=out_path,
                                           out_file="train_spatial.nc",
                                           lat1=area_dic[area]["lat1"],
                                           lat2=area_dic[area]["lat2"],
                                           lon1=area_dic[area]["lon1"],
                                           lon2=area_dic[area]["lon2"],
                                           area_name=area,
                                           doy=pair[0],
                                           n_hist=30,
                                           selected_sentinel_fields=["soil_moisture", "tb_v_disaggregated"])
                plot_single_variable(out_path, ["train_spatial"], "soil_moisture", out_path, "usa", "cm**3/cm**3",
                                     v_min=0.0, v_max=v_max)
                convert2csv(out_path, "train_spatial", doy=pair[0])

                train_spatial_mask = get_variable_mask(os.path.join("Data", "Sentinel", "usa_states_real_gap", "mask_by_state",
                                                                    area, pair[0]+".nc"),
                                                       "soil_moisture")
                merge_various_variables_mask_v2(mask_array=~train_spatial_mask,
                                                mask_cut=True,
                                                out_path=out_path,
                                                out_file="test.nc",
                                                lat1=area_dic[area]["lat1"],
                                                lat2=area_dic[area]["lat2"],
                                                lon1=area_dic[area]["lon1"],
                                                lon2=area_dic[area]["lon2"],
                                                area_name=area,
                                                doy=pair[0],
                                                n_hist=30,
                                                selected_sentinel_fields=["soil_moisture", "tb_v_disaggregated"],
                                                state_mask=True,
                                                state=area)
                plot_single_variable(out_path, ["test"], "smap_p_e_soil_moisture", out_path, "usa", "cm**3/cm**3",
                                     v_min=0.0, v_max=v_max)
                convert2csv(out_path, "test", doy=pair[0])

                test_mask = get_variable_mask(os.path.join(out_path, "test.nc"), "smap_p_e_soil_moisture")
                for temporal_doy in pair[1:]:
                    if "train_temporal_" + temporal_doy + ".nc" in os.listdir(os.path.join(out_path, "train_temporal")):
                        merge_various_variables_mask_v2(mask_array=test_mask,
                                                        mask_cut=False,
                                                        out_path=get_out_path(os.path.join(out_path, "train_temporal")),
                                                        out_file="train_temporal_" + temporal_doy + ".nc",
                                                        lat1=area_dic[area]["lat1"],
                                                        lat2=area_dic[area]["lat2"],
                                                        lon1=area_dic[area]["lon1"],
                                                        lon2=area_dic[area]["lon2"],
                                                        area_name=area,
                                                        doy=temporal_doy,
                                                        n_hist=30,
                                                        selected_sentinel_fields=["soil_moisture", "tb_v_disaggregated"],
                                                        sentinel_path=os.path.join("Data", "Sentinel", "usa_states_real_gap",
                                                                                   "mask_by_state", area))
                        plot_single_variable(os.path.join(out_path, "train_temporal"), ["train_temporal_" + temporal_doy],
                                             "soil_moisture", os.path.join(out_path, "train_temporal"), "usa",
                                             "cm**3/cm**3", v_min=0.0, v_max=v_max)
                        convert2csv(os.path.join(out_path, "train_temporal"), "train_temporal_" + temporal_doy,
                                    doy=temporal_doy)

                merge_csv_files(in_path=os.path.join(out_path, "train_temporal"),
                                out_path=out_path,
                                out_file="train_temporal.csv")

                merge_csv_files(in_path=out_path,
                                out_path=out_path,
                                out_file="train_spatial_temporal.csv",
                                in_file_lis=["train_spatial.csv", "train_temporal.csv"])

    real_gap_area_pair_dic = load_pkl("real_gap_most_recent_pairs.pkl")
    for area in ["arizona", "oklahoma", "iowa"]:
        for pair in real_gap_area_pair_dic[area]:
            if pair[0] in os.listdir(os.path.join("real_gap_sttl_experiment", area)):
                v_max = 0.2 if area == "arizona" else 0.5
                out_path = get_out_path(os.path.join("real_gap_sttl_experiment", area, pair[0]))

                merge_various_variables_v2(sentinel_path=os.path.join("Data", "Sentinel", "usa_states_real_gap",
                                                                      "mask_by_state", area),
                                           out_path=out_path,
                                           out_file="train_spatial.nc",
                                           lat1=area_dic[area]["lat1"],
                                           lat2=area_dic[area]["lat2"],
                                           lon1=area_dic[area]["lon1"],
                                           lon2=area_dic[area]["lon2"],
                                           area_name=area,
                                           doy=pair[0],
                                           n_hist=30,
                                           selected_sentinel_fields=["soil_moisture", "tb_v_disaggregated"])
                plot_single_variable(out_path, ["train_spatial"], "soil_moisture", out_path, "usa", "cm**3/cm**3",
                                     v_min=0.0, v_max=v_max)
                convert2csv(out_path, "train_spatial", doy=pair[0])

                train_spatial_mask = get_variable_mask(os.path.join("Data", "Sentinel", "usa_states_real_gap", "mask_by_state",
                                                                    area, pair[0]+".nc"),
                                                       "soil_moisture")
                merge_various_variables_mask_v2(mask_array=~train_spatial_mask,
                                                mask_cut=True,
                                                out_path=out_path,
                                                out_file="test.nc",
                                                lat1=area_dic[area]["lat1"],
                                                lat2=area_dic[area]["lat2"],
                                                lon1=area_dic[area]["lon1"],
                                                lon2=area_dic[area]["lon2"],
                                                area_name=area,
                                                doy=pair[0],
                                                n_hist=30,
                                                selected_sentinel_fields=["soil_moisture", "tb_v_disaggregated"],
                                                state_mask=True,
                                                state=area)
                plot_single_variable(out_path, ["test"], "smap_p_e_soil_moisture", out_path, "usa", "cm**3/cm**3",
                                     v_min=0.0, v_max=v_max)
                convert2csv(out_path, "test", doy=pair[0])

                test_mask = get_variable_mask(os.path.join(out_path, "test.nc"), "smap_p_e_soil_moisture")
                for temporal_doy in pair[1:]:
                    if "train_temporal_" + temporal_doy + ".nc" in os.listdir(os.path.join(out_path, "train_temporal")):
                        merge_various_variables_mask_v2(mask_array=test_mask,
                                                        mask_cut=False,
                                                        out_path=get_out_path(os.path.join(out_path, "train_temporal")),
                                                        out_file="train_temporal_" + temporal_doy + ".nc",
                                                        lat1=area_dic[area]["lat1"],
                                                        lat2=area_dic[area]["lat2"],
                                                        lon1=area_dic[area]["lon1"],
                                                        lon2=area_dic[area]["lon2"],
                                                        area_name=area,
                                                        doy=temporal_doy,
                                                        n_hist=30,
                                                        selected_sentinel_fields=["soil_moisture", "tb_v_disaggregated"],
                                                        sentinel_path=os.path.join("Data", "Sentinel", "usa_states_real_gap",
                                                                                   "mask_by_state", area))
                        plot_single_variable(os.path.join(out_path, "train_temporal"), ["train_temporal_" + temporal_doy],
                                             "soil_moisture", os.path.join(out_path, "train_temporal"), "usa",
                                             "cm**3/cm**3", v_min=0.0, v_max=v_max)
                        convert2csv(os.path.join(out_path, "train_temporal"), "train_temporal_" + temporal_doy,
                                    doy=temporal_doy)

                merge_csv_files(in_path=os.path.join(out_path, "train_temporal"),
                                out_path=out_path,
                                out_file="train_temporal.csv")

                merge_csv_files(in_path=out_path,
                                out_path=out_path,
                                out_file="train_spatial_temporal.csv",
                                in_file_lis=["train_spatial.csv", "train_temporal.csv"])


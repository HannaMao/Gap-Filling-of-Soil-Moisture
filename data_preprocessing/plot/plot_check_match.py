# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause
from data_preprocessing import index_states_dic
from ..utils import get_out_path
from . import plot_single_variable


import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_temporal_based_match():
    for v_name in ["sm", "tb"]:
        plot_single_variable(os.path.join("Data", "Analysis", "check_match", "temporal_based"),
                             ["usa_20171105_20180315"],
                             "_".join([v_name, "corr"]),
                             os.path.join("Plots", "Analysis", "check_match", "temporal_based",
                                          "usa_20171105_20180315"),
                             "usa",
                             unit="", v_min=0, v_max=1)
        plot_single_variable(os.path.join("Data", "Analysis", "check_match", "temporal_based"),
                             ["usa_20171105_20180315"],
                             "_".join([v_name, "r2"]),
                             os.path.join("Plots", "Analysis", "check_match", "temporal_based",
                                          "usa_20171105_20180315"),
                             "usa",
                             unit="", v_min=0, v_max=1)
        plot_single_variable(os.path.join("Data", "Analysis", "check_match", "temporal_based"),
                             ["usa_20171105_20180315"],
                             "_".join([v_name, "rmse"]),
                             os.path.join("Plots", "Analysis", "check_match", "temporal_based",
                                          "usa_20171105_20180315"),
                             "usa",
                             unit="")
        plot_single_variable(os.path.join("Data", "Analysis", "check_match", "temporal_based"),
                             ["usa_20171105_20180315"],
                             "_".join([v_name, "bias"]),
                             os.path.join("Plots", "Analysis", "check_match", "temporal_based",
                                          "usa_20171105_20180315"),
                             "usa",
                             unit="", v_min=-0.1, v_max=0.1)


def plot_match_2d(in_path, out_path):
    for nc_file in os.listdir(in_path):
        if nc_file.endswith(".nc"):
            for v_name in ["sm", "tb"]:
                oout_path = os.path.join(out_path, v_name + "_corr") if "_" not in nc_file else out_path
                plot_single_variable(in_path,
                                     [nc_file[:-3]],
                                     "_".join([v_name, "corr"]),
                                     oout_path,
                                     "usa",
                                     unit="", v_min=0, v_max=1)
                oout_path = os.path.join(out_path, v_name + "_r2") if "_" not in nc_file else out_path
                plot_single_variable(in_path,
                                     [nc_file[:-3]],
                                     "_".join([v_name, "r2"]),
                                     oout_path,
                                     "usa",
                                     unit="", v_min=0, v_max=1)
                oout_path = os.path.join(out_path, v_name + "_rmse") if "_" not in nc_file else out_path
                v_min = 0.0 if v_name == "sm" else 0.0
                v_max = 0.25 if v_name == "sm" else 20.0
                plot_single_variable(in_path,
                                     [nc_file[:-3]],
                                     "_".join([v_name, "rmse"]),
                                     oout_path,
                                     "usa",
                                     unit="", v_min=v_min, v_max=v_max)
                oout_path = os.path.join(out_path, v_name + "_bias") if "_" not in nc_file else out_path
                v_min = -0.05 if v_name == "sm" else -0.5
                v_max = 0.05 if v_name == "sm" else 0.5
                plot_single_variable(in_path,
                                     [nc_file[:-3]],
                                     "_".join([v_name, "bias"]),
                                     oout_path,
                                     "usa",
                                     unit="", v_min=v_min, v_max=v_max)
            if "_" in nc_file:
                plot_single_variable(in_path,
                                     [nc_file[:-3]],
                                     "n_grids",
                                     out_path,
                                     "usa",
                                     unit="")


def plot_states_based_match_2d(min_grids):
    # in_path = os.path.join("Data", "Analysis", "check_match", "states_based", str(min_grids))
    # out_path = os.path.join("Data_Plots", "Analysis", "check_match", "states_based", str(min_grids), "2d")
    # plot_match_2d(in_path, out_path)

    in_path = os.path.join("Data", "Analysis", "check_match", "states_based")
    out_path = os.path.join("Data_Plots", "Analysis", "check_match", "states_based")
    plot_match_2d(in_path, out_path)


def plot_sequence(state, f_name, values, doy_values, grids_values, min_grids):
    out_path = get_out_path(os.path.join("Data_Plots", "Analysis", "check_match", "states_based", str(min_grids), "sequence",
                                         f_name))
    name_dic = {"sm": "Soil Moisture", "tb": "Brightness Temperature",
                "r2": "R2", "rmse": "RMSE", "bias": "Bias", "corr": "Correlation"}

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    x = range(len(values))
    color = ["C3" if _ > min_grids else "C7" for _ in map(int, grids_values)]
    ax.scatter(x, values, marker="s", color=color)
    ax.plot(x, values)

    ax.set_xlabel("DOY")
    ax.set_ylabel(name_dic[f_name.split("_")[1]])
    for a, b, c in zip(x, values, grids_values):
        plt.text(a, b, "{0:.4f}\n{1}".format(b, c), horizontalalignment="right")
    plt.xticks(x, doy_values, rotation='vertical')
    plt.title(state.capitalize() + " " + name_dic[f_name.split("_")[0]] + " " + name_dic[f_name.split("_")[1]])
    plt.savefig(os.path.join(out_path, state + ".jpg"))
    plt.close()


def plot_states_based_match_sequence(min_grids):
    in_path = os.path.join("Data", "Analysis", "check_match", "states_based", str(min_grids), "statistics")

    all_states = [index_states_dic[key] for key in sorted(list(index_states_dic.keys()))]
    results = {}
    for v_name in ["sm", "tb"]:
        for stat in ["r2", "rmse", "corr", "bias"]:
            f_name = "_".join([v_name, stat])
            results[f_name] = pd.read_csv(os.path.join(in_path, f_name + ".csv"))
    results["n_grids"] = pd.read_csv(os.path.join(in_path, "n_grids.csv"))

    for state in all_states:
        grids_values = [x for x in results["n_grids"][state] if x != "UNK"]
        if len(grids_values) > 0:
            doy_values = [results["n_grids"]["DOY"][i] for i, x in enumerate(results["n_grids"][state]) if x != "UNK"]

            for v_name in ["sm", "tb"]:
                for stat in ["r2", "rmse", "corr", "bias"]:
                    f_name = "_".join([v_name, stat])
                    values = [*map(float, [x for x in results[f_name][state] if x != "UNK"])]
                    plot_sequence(state, f_name, values, doy_values, grids_values, min_grids)


def plot_sequence_comparisons(state, f_name, sm_values, tb_values, doy_values, grids_values, min_grids):
    out_path = get_out_path(
        os.path.join("Data_Plots", "Analysis", "check_match", "states_based", str(min_grids), "sequence",
                     f_name))
    name_dic = {"sm": "Soil Moisture", "tb": "Brightness Temperature",
                "r2": "R2", "rmse": "RMSE", "bias": "Bias", "corr": "Correlation"}

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    x = range(len(sm_values))
    ax.scatter(x, sm_values, marker="s", color="C3")
    ax.scatter(x, tb_values, marker="x", color="C4")
    ax.plot(x, sm_values, "C0")
    ax.plot(x, tb_values, "C1")

    ax.set_xlabel("DOY")
    ax.set_ylabel(name_dic[f_name.split("_")[0]])
    for a, b, c in zip(x, sm_values, grids_values):
        plt.text(a, b, "{0:.4f}\n{1}".format(b, c), horizontalalignment="right")
    for a, b, c in zip(x, tb_values, grids_values):
        plt.text(a, b, "{0:.4f}\n{1}".format(b, c), horizontalalignment="right")
    plt.xticks(x, doy_values, rotation='vertical')
    plt.title(state.capitalize() + " " + name_dic[f_name.split("_")[0]] + " Comparison")
    plt.savefig(os.path.join(out_path, state + ".jpg"))
    plt.close()


def plot_states_based_match_sequence_comparisons(min_grids):
    in_path = os.path.join("Data", "Analysis", "check_match", "states_based", str(min_grids), "statistics")

    all_states = [index_states_dic[key] for key in sorted(list(index_states_dic.keys()))]
    results = {}
    for v_name in ["sm", "tb"]:
        for stat in ["r2", "rmse", "corr", "bias"]:
            f_name = "_".join([v_name, stat])
            results[f_name] = pd.read_csv(os.path.join(in_path, f_name + ".csv"))
    results["n_grids"] = pd.read_csv(os.path.join(in_path, "n_grids.csv"))

    for state in all_states:
        grids_values = [x for x in results["n_grids"][state] if x != "UNK"]
        if len(grids_values) > 0:
            doy_values = [results["n_grids"]["DOY"][i] for i, x in enumerate(results["n_grids"][state]) if x != "UNK"]

            for stat in ["r2", "rmse", "corr", "bias"]:
                f_name = "_".join([stat, "comparison"])
                sm_values = [*map(float, [x for x in results["sm_"+stat][state] if x != "UNK"])]
                tb_values = [*map(float, [x for x in results["tb_"+stat][state] if x != "UNK"])]
                plot_sequence_comparisons(state, f_name, sm_values, tb_values, doy_values, grids_values, min_grids)


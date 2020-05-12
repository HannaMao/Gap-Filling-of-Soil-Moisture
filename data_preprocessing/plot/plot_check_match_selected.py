# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..utils import get_out_path

import os
import pandas as pd
import matplotlib.pyplot as plt


name_dic = {"sm": "Soil Moisture", "tb": "Brightness Temperature",
            "r2": "R2", "rmse": "RMSE", "bias": "Bias", "corr": "Correlation"}


def _plot_sequence_comparisons(out_path, area, f_name, sm_values, tb_values, doy_values, grids_values):
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
    plt.title(area.capitalize() + " " + name_dic[f_name.split("_")[0]] + " Comparison")
    plt.savefig(os.path.join(out_path, f_name + ".jpg"))
    plt.close()


def _plot_sequence(out_path, area, f_name, values, doy_values, grids_values):
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    x = range(len(values))
    ax.scatter(x, values, marker="s")
    ax.plot(x, values)

    ax.set_xlabel("DOY")
    ax.set_ylabel(name_dic[f_name.split("_")[1]])
    for a, b, c in zip(x, values, grids_values):
        plt.text(a, b, "{0:.4f}\n{1}".format(b, c), horizontalalignment="right")
    plt.xticks(x, doy_values, rotation='vertical')
    plt.title(area.capitalize() + " " + name_dic[f_name.split("_")[0]] + " " + name_dic[f_name.split("_")[1]])
    plt.savefig(os.path.join(out_path, f_name + ".jpg"))
    plt.close()


def plot_sequence_selected(area):
    results = pd.read_csv(os.path.join("Data", "Analysis", "check_match", "selected", area + ".csv"))
    out_path = get_out_path(os.path.join("Data_Plots", "Analysis", "check_match", "selected", area))

    for stat in ["r2", "rmse", "corr", "bias"]:
        f_name = "_".join([stat, "comparison"])
        _plot_sequence_comparisons(out_path,
                                   area,
                                   f_name,
                                   results["sm_" + stat],
                                   results["tb_" + stat],
                                   results["DOY"],
                                   results["n_grids"])
        for v_name in ["sm", "tb"]:
            f_name = "_".join([v_name, stat])
            _plot_sequence(out_path,
                           area,
                           f_name,
                           results[f_name],
                           results["DOY"],
                           results["n_grids"])




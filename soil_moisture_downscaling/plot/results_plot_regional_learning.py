# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from soil_moisture_downscaling.utils import get_out_path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

stat_abbr = {"corr": "Corr", "r2": "R2", "rmse": "RMSE", "ubrmse": "ubRMSE"}
stat_name = {"corr": "Correlation", "r2": "R2", "rmse": "RMSE", "ubrmse": "ubRMSE"}


def plot_comparison_before_after(results, out_path, stat, d_type):
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)

    values = np.asarray([results["_".join([stat_abbr[stat], d_type, "before"])][0],
                         results["_".join([stat_abbr[stat], d_type])][0]])
    labels = ["before", "after"]
    plt.bar(np.arange(2), values)
    plt.ylabel(stat_name[stat])
    plt.xticks(np.arange(2), labels)
    ax.grid()
    plt.savefig(os.path.join(out_path, "_".join([stat, d_type]) + ".jpg"))
    plt.close()


def plot_comparison_train_test(results, out_path, stat, step):
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    values = []
    var_name = "_".join([stat_abbr[stat], "train", step]) if step == "before" else \
        "_".join([stat_abbr[stat], "train"])
    values.append(results[var_name][0])
    var_name = "_".join([stat_abbr[stat], "test", step]) if step == "before" else \
        "_".join([stat_abbr[stat], "test"])
    values.append(results[var_name][0])
    labels = ["train", "test"]

    plt.bar(np.arange(2), np.asarray(values))
    plt.ylabel(stat_name[stat])
    plt.xticks(np.arange(2), labels)
    plt.savefig(os.path.join(out_path, "_".join([step, stat]) + ".jpg"))
    plt.close()


def results_plot_regional_learning(in_path):
    results = pd.read_csv(os.path.join(in_path, "results.csv"))

    plot_comparison_before_after(results, get_out_path(os.path.join(in_path, "plots")), "r2", "train")
    plot_comparison_before_after(results, get_out_path(os.path.join(in_path, "plots")), "corr", "train")
    plot_comparison_before_after(results, get_out_path(os.path.join(in_path, "plots")), "rmse", "train")
    plot_comparison_before_after(results, get_out_path(os.path.join(in_path, "plots")), "ubrmse", "train")

    plot_comparison_before_after(results, get_out_path(os.path.join(in_path, "plots")), "r2", "test")
    plot_comparison_before_after(results, get_out_path(os.path.join(in_path, "plots")), "corr", "test")
    plot_comparison_before_after(results, get_out_path(os.path.join(in_path, "plots")), "rmse", "test")
    plot_comparison_before_after(results, get_out_path(os.path.join(in_path, "plots")), "ubrmse", "test")

    plot_comparison_train_test(results, get_out_path(os.path.join(in_path, "plots")), "r2", "before")
    plot_comparison_train_test(results, get_out_path(os.path.join(in_path, "plots")), "r2", "after")

    plot_comparison_train_test(results, get_out_path(os.path.join(in_path, "plots")), "corr", "before")
    plot_comparison_train_test(results, get_out_path(os.path.join(in_path, "plots")), "corr", "after")

    plot_comparison_train_test(results, get_out_path(os.path.join(in_path, "plots")), "rmse", "before")
    plot_comparison_train_test(results, get_out_path(os.path.join(in_path, "plots")), "rmse", "after")

    plot_comparison_train_test(results, get_out_path(os.path.join(in_path, "plots")), "ubrmse", "before")
    plot_comparison_train_test(results, get_out_path(os.path.join(in_path, "plots")), "ubrmse", "after")













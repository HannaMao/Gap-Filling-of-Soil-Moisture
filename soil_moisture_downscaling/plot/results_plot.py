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


def plot_numbers(results, out_path):
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    x = range(len(results.index))
    ax.plot(x, np.array(results["Number_train"]), "b--", label="Number of Train Tuples")
    ax.plot(x, np.array(results["Number_test"]), "r--", label="Number of Test Tuples")
    ax.legend(loc='best', shadow=True)
    ax.set_xlabel("Shrinking Width (km)")
    ax.set_ylabel("Number")
    plt.xticks(x, results["File"])
    plt.savefig(os.path.join(out_path, "Number.jpg"))
    plt.close()


def plot_comparison_before_after(results, out_path, stat, d_type):
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    x = range(len(results.index))
    ax.plot(x, np.array(results["_".join([stat_abbr[stat], d_type, "before"])]), "b--",
            label=stat_name[stat] + " of " + d_type.capitalize() + " before ML")
    ax.plot(x, np.array(results["_".join([stat_abbr[stat], d_type])]), "r--",
            label=stat_name[stat] + " of " + d_type.capitalize() + " after ML")
    ax.legend(loc='best', shadow=True)
    ax.set_xlabel("Shrinking Width (km)")
    ax.set_ylabel(stat_name[stat])
    ax.grid()
    plt.xticks(x, results["File"])
    plt.savefig(os.path.join(out_path, "_".join([stat, d_type]) + ".jpg"))
    plt.close()


def plot_comparison_train_test(results, out_path, stat, step):
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    x = range(len(results.index))
    var_name = "_".join([stat_abbr[stat], "train", step]) if step == "before" else \
        "_".join([stat_abbr[stat], "train"])
    ax.plot(x, np.array(results[var_name]), "b--",
            label=stat_name[stat] + " of Train " + step + " ML")
    var_name = "_".join([stat_abbr[stat], "test", step]) if step == "before" else \
        "_".join([stat_abbr[stat], "test"])
    ax.plot(x, np.array(results[var_name]), "r--",
            label=stat_name[stat] + " of Test " + step + " ML")
    ax.legend(loc='best', shadow=True)
    ax.set_xlabel("Shrinking Width (km)")
    ax.set_ylabel(stat_name[stat])
    ax.grid()
    plt.xticks(x, results["File"])
    plt.savefig(os.path.join(out_path, "_".join([step, stat]) + ".jpg"))
    plt.close()


def results_plot(in_path):
    results = pd.read_csv(os.path.join(in_path, "results.csv"))

    plot_numbers(results, get_out_path(os.path.join(in_path, "plots")))

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













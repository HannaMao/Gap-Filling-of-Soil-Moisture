# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import os
import csv
import numpy as np


def get_lat_lon(reso):
    path = os.path.join("Data", "EASE2_Grids")
    for f in os.listdir(path):
        if f.startswith(reso):
            lats, lons = [], []
            for ff in os.listdir(os.path.join(path, f)):
                if "lats" in ff and ff.endswith(".csv"):
                    lats = np.genfromtxt(os.path.join(path, f, ff), delimiter=",")
                elif "lons" in ff and ff.endswith(".csv"):
                    lons = np.genfromtxt(os.path.join(path, f, ff), delimiter=",")

            return lats, lons


def match_lat_lon(lats_from, lons_from, lats_to, lons_to, expand=0):
    i_lat_start = i_lat_end = i_lon_start = i_lon_end = 0

    for i in range(len(lats_from)):
        if abs(lats_from[i] - lats_to[0]) < 0.00001:
            i_lat_start = i - expand
        if abs(lats_from[i] - lats_to[-1]) < 0.00001:
            i_lat_end = i + expand
    for i in range(len(lons_from)):
        if abs(lons_from[i] - lons_to[0]) < 0.00001:
            i_lon_start = i - expand
        if abs(lons_from[i] - lons_to[-1]) < 0.00001:
            i_lon_end = i + expand

    return i_lat_start, i_lat_end, i_lon_start, i_lon_end

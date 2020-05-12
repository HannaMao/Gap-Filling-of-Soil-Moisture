# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import os
import numpy as np
import csv


def convert_double_to_csv():
    grid_path = os.path.join("Data", "EASE2_Grids")
    for reso in os.listdir(grid_path):
        if "M" in reso:
            reso_path = os.path.join(grid_path, reso)
            for double_file in os.listdir(reso_path):
                if ".double" in double_file:
                    n_lon, n_lat = map(int, double_file.split(".")[2].split("x")[:2])
                    f_name = double_file.split(".")[1]
                    data2d = np.fromfile(os.path.join(reso_path, double_file), dtype=np.float64).reshape((n_lat, n_lon))
                    if f_name == "lats":
                        data1d = data2d.mean(axis=1)
                    elif f_name == "lons":
                        data1d = data2d.mean(axis=0)
                    np.savetxt(os.path.join(reso_path, f_name + ".csv"), data1d, delimiter=",")

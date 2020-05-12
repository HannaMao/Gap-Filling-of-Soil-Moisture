# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
from operator import or_


def generate_mask(file_in, v_name, folder_out):
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    usa_file = Dataset(file_in, "r")

    n_lat = usa_file.dimensions["lat"].size
    n_lon = usa_file.dimensions["lon"].size
    v_flag = ma.getmaskarray(usa_file.variables[v_name][:])

    lat_3km_usa, lon_3km_usa = [], []

    for i in range(n_lat):
        for j in range(n_lon):
            if not v_flag[i][j]:
                lat_3km_usa.append(i)
                lon_3km_usa.append(j)

    lat_3km_global = [lat_index + 566 for lat_index in lat_3km_usa]
    lon_3km_global = [lon_index + 1767 for lon_index in lon_3km_usa]

    global_9km = set()
    for lat_index, lon_index in zip(lat_3km_global, lon_3km_global):
        global_9km.add((lat_index//3, lon_index//3))

    lat_9km_global = [lat_lon[0] for lat_lon in global_9km]
    lon_9km_global = [lat_lon[1] for lat_lon in global_9km]
    lat_9km_usa = [lat_index - 171 for lat_index in lat_9km_global]
    lon_9km_usa = [lon_index - 568 for lon_index in lon_9km_global]

    print(len(lat_3km_usa), len(lon_3km_usa))
    print(len(lat_3km_global), len(lon_3km_global))
    print(len(lat_9km_global), len(lon_9km_global))
    print(len(lat_9km_usa), len(lon_9km_usa))

    np.savetxt(os.path.join(folder_out, "lat_3km_usa.csv"), np.asarray(lat_3km_usa))
    np.savetxt(os.path.join(folder_out, "lon_3km_usa.csv"), np.asarray(lon_3km_usa))
    np.savetxt(os.path.join(folder_out, "lat_3km_global.csv"), np.asarray(lat_3km_global))
    np.savetxt(os.path.join(folder_out, "lon_3km_global.csv"), np.asarray(lon_3km_global))
    np.savetxt(os.path.join(folder_out, "lat_9km_global.csv"), np.asarray(lat_9km_global))
    np.savetxt(os.path.join(folder_out, "lon_9km_global.csv"), np.asarray(lon_9km_global))
    np.savetxt(os.path.join(folder_out, "lat_9km_usa.csv"), np.asarray(lat_9km_usa))
    np.savetxt(os.path.join(folder_out, "lon_9km_usa.csv"), np.asarray(lon_9km_usa))


def generate_numpy_mask(folder_in, lat_file, lon_file, n_lat, n_lon):
    mask_array = np.ones((n_lat, n_lon), dtype=bool)
    lat_indices = map(int, np.loadtxt(os.path.join(folder_in, lat_file)))
    lon_indices = map(int, np.loadtxt(os.path.join(folder_in, lon_file)))

    for lat_i, lon_i in zip(lat_indices, lon_indices):
        if 0 <= lat_i <= n_lat - 1 and 0 <= lon_i <= n_lon - 1:
            mask_array[lat_i, lon_i] = False

    return mask_array


def get_variable_mask(in_file, variable):
    fh_in = Dataset(in_file, "r")
    return ma.getmaskarray(fh_in.variables[variable][:])


def get_two_variables_mask(file_1, file_2, variable1, variable2):
    fh_1 = Dataset(file_1, "r")
    fh_2 = Dataset(file_2, "r")
    return np.logical_or.reduce([ma.getmaskarray(fh_1.variables[variable1][:]),
                                 ma.getmaskarray(fh_2.variables[variable2][:])])






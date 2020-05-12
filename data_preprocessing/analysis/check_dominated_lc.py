# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from data_preprocessing.utils import select_area, get_lat_lon, match_lat_lon
from data_preprocessing import landcover_class_dic

import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma


def check_dominated_lc(lat1, lat2, lon1, lon2, reso):
    fh = Dataset(os.path.join("Data", "LANDCOVER", "landcover_" + reso + "_usa_2016.nc"), "r")
    n_dim = str(int(reso[:-2])).zfill(2)
    lat_indices, lon_indices = select_area(lat1, lat2, lon1, lon2, "M" + n_dim)
    lats, lons = get_lat_lon("M" + n_dim)
    assert (len(lats) != 0 and len(lons) != 0)
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]

    i_lat_start, i_lat_end, i_lon_start, i_lon_end = match_lat_lon(fh.variables["lat"][:],
                                                                   fh.variables["lon"][:],
                                                                   lats,
                                                                   lons)

    selected = fh.variables["lc_1"][i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1].compressed()
    lc_id, lc_count = np.unique(selected, return_counts=True)
    dominated_lc_ids, dominated_lc_names = [], []
    for i, c in zip(lc_id, lc_count):
        if c / len(selected) > 0.1:
            dominated_lc_ids.append(i), dominated_lc_names.append(landcover_class_dic[int(i)])

    return dominated_lc_ids, dominated_lc_names


def check_dominated_lc_indices(lat_s, lat_e, lon_s, lon_e, reso):
    fh = Dataset(os.path.join("Data", "LANDCOVER", "landcover_" + reso + "_usa_2016.nc"), "r")

    selected = fh.variables["lc_1"][lat_s: lat_e, lon_s: lon_e].compressed()
    lc_id, lc_count = np.unique(selected, return_counts=True)
    dominated_lc_ids, dominated_lc_names = [], []
    for i, c in zip(lc_id, lc_count):
        if c / len(selected) > 0.01:
            dominated_lc_ids.append(i), dominated_lc_names.append(landcover_class_dic[int(i)])

    return dominated_lc_ids, dominated_lc_names


def check_dominated_lc_file(in_file, v_name, reso):
    fh = Dataset(os.path.join("Data", "LANDCOVER", "landcover_" + reso + "_usa_2016.nc"), "r")
    fh_mask = Dataset(in_file, "r")

    selected = ma.array(fh.variables["lc_1"][:], mask=ma.getmaskarray(fh_mask.variables[v_name][:])).compressed()
    lc_id, lc_count = np.unique(selected, return_counts=True)
    dominated_lc_ids, dominated_lc_names = [], []
    for i, c in zip(lc_id, lc_count):
        if c / len(selected) > 0.01:
            dominated_lc_ids.append(i), dominated_lc_names.append(landcover_class_dic[int(i)])

    return dominated_lc_ids, dominated_lc_names

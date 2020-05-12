# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..preprocess import modis_lst_fill_missing_by_most_recent_values
from ..utils import select_area, get_lat_lon, match_lat_lon, get_out_path
from ..utils import Logger

import datetime
import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import sys


def generate_nearly_covered_modis_lst(area, doy, lat1, lat2, lon1, lon2):
    in_path = os.path.join("Data", "MOD11A1", "3km")
    out_path = get_out_path(os.path.join("Data", "MOD11A1", "3km_nearly_overlapped"))
    log_file = os.path.join(out_path, 'log.txt')
    sys.stdout = Logger(log_file)

    print(area, doy)
    if doy + ".nc" in os.listdir(os.path.join("Data", "Sentinel", "usa_filtered")):
        fh_sentinel = Dataset(os.path.join("Data", "Sentinel", "usa_filtered", doy + ".nc"), "r")
    elif doy + ".nc" in os.listdir(os.path.join("Data", "Sentinel", "usa_filtered_real_gap")):
        fh_sentinel = Dataset(os.path.join("Data", "Sentinel", "usa_filtered_real_gap", doy + ".nc"), "r")

    lat_indices, lon_indices = select_area(lat1, lat2, lon1, lon2, "M03")
    lats, lons = get_lat_lon("M03")
    assert (len(lats) != 0 and len(lons) != 0)
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]

    i_lat_start, i_lat_end, i_lon_start, i_lon_end = match_lat_lon(fh_sentinel.variables["lat"][:],
                                                                   fh_sentinel.variables["lon"][:],
                                                                   lats,
                                                                   lons)
    sm_mask = ~ma.getmaskarray(fh_sentinel.variables["soil_moisture"]
                              [i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1])
    if np.sum(sm_mask) > 0:
        s_day = datetime.datetime.strptime(doy, "%Y%m%d").date()
        n_days = 0
        lst_day_ref = []
        while True:
            hist_day = str(s_day + datetime.timedelta(days=-n_days)).replace("-", "")
            print(hist_day,)

            fh = Dataset(os.path.join(in_path, hist_day + ".nc"), "r")
            lst_day_ref.append(fh.variables["LST_Day"][i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1])
            fh.close()

            lst_day_ref_mask = ~np.all(ma.getmaskarray(ma.asarray(lst_day_ref)), axis=0)
            overlap = np.logical_and(sm_mask, lst_day_ref_mask)
            if np.sum(overlap) / float(np.sum(sm_mask)) > 0.95:
                modis_lst_fill_missing_by_most_recent_values(doy, area, n_days)
                break

            n_days += 1


def generate_nearly_covered_modis_lst_usa(doy):
    in_path = os.path.join("Data", "MOD11A1", "3km")
    out_path = get_out_path(os.path.join("Data", "MOD11A1", "3km_nearly_overlapped"))
    log_file = os.path.join(out_path, 'log.txt')
    sys.stdout = Logger(log_file)

    print(doy)
    fh_sentinel = Dataset(os.path.join("Data", "Sentinel", "usa_db", doy + ".nc"), "r")
    sm_mask = ~ma.getmaskarray(fh_sentinel.variables["soil_moisture"][:])

    s_day = datetime.datetime.strptime(doy, "%Y%m%d").date()
    n_days = 0
    lst_day_ref = []
    while True:
        hist_day = str(s_day + datetime.timedelta(days=-n_days)).replace("-", "")
        print(hist_day,)

        fh = Dataset(os.path.join(in_path, hist_day + ".nc"), "r")
        lst_day_ref.append(fh.variables["LST_Day"][:])
        fh.close()

        lst_day_ref_mask = ~np.all(ma.getmaskarray(ma.asarray(lst_day_ref)), axis=0)
        overlap = np.logical_and(sm_mask, lst_day_ref_mask)

        if np.sum(overlap) / float(np.sum(sm_mask)) > 0.95:
            print(n_days)
            modis_lst_fill_missing_by_most_recent_values(doy, "usa", n_days)
            break

        n_days += 1





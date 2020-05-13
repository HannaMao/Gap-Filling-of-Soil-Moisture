# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..utils import select_area, get_lat_lon, match_lat_lon, round_sigfigs, vprint

import os
from netCDF4 import Dataset
import numpy.ma as ma
import numpy as np
from sklearn.metrics import r2_score
from datetime import datetime
from collections import defaultdict


def historical_values_search(data_path, f_name, search_path, lat1, lat2, lon1, lon2, doy, var_list, out_path,
                             output, verbose,
                             n_threshold=50, r2_threshold=0.6):
    vprint(" ".join([data_path, f_name]), verbose)
    fh_in = dict()
    fh_in["train"] = Dataset(os.path.join(data_path, f_name + "_train.nc"), "r")
    fh_in["test"] = Dataset(os.path.join(data_path, f_name + "_test.nc"), "r")
    doys = sorted([x[:8] for x in os.listdir(search_path) if x.endswith(".nc")])
    doy_index = doys.index(doy)

    lat_indices, lon_indices = select_area(lat1, lat2, lon1, lon2, "M03")
    lats, lons = get_lat_lon("M03")
    assert (len(lats) != 0 and len(lons) != 0)
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]
    fh_usa = Dataset(os.path.join(search_path, doys[0] + ".nc"), "r")
    i_lat_start, i_lat_end, i_lon_start, i_lon_end = match_lat_lon(fh_usa.variables["lat"][:],
                                                                   fh_usa.variables["lon"][:],
                                                                   lats,
                                                                   lons)
    mask_var_train = ma.getmaskarray(fh_in["train"].variables[var_list[0]][:])
    mask_var_test = ma.getmaskarray(fh_in["test"].variables[var_list[0]][:])

    first_flag = True
    return_r2 = {}
    abandoned_r2 = {}
    for var in var_list:
        return_r2[var] = defaultdict(list)
        abandoned_r2[var] = defaultdict(list)
    for d in doys[:doy_index]:
        mask_or = {}
        fh_comp = Dataset(os.path.join(search_path, d + ".nc"), "r")
        mask_or["train"] = [mask_var_train]
        mask_or["test"] = [mask_var_test]
        for var in var_list:
            v_comp = fh_comp.variables[var][i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1]
            mask_comp = ma.getmaskarray(v_comp)
            mask_or["train"].append(mask_comp)
            mask_or["test"].append(mask_comp)
        mask_or["train"] = np.logical_or.reduce(mask_or["train"])
        mask_or["test"] = np.logical_or.reduce(mask_or["test"])
        if np.bitwise_not(mask_or["train"]).sum() > n_threshold and np.bitwise_not(mask_or["test"]).sum() > n_threshold:
            for var in var_list:
                varin = fh_comp.variables[var]
                r2_train = r2_score(ma.array(fh_in["train"].variables[var][:],
                                             mask=mask_or["train"]).compressed(),
                                    ma.array(varin[i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1],
                                             mask=mask_or["train"]).compressed())
                if r2_train > r2_threshold:
                    if first_flag and output:
                        fh_out = {}
                        for f_type in ["train", "test"]:
                            fh_out[f_type] = Dataset(os.path.join(out_path, "_".join([f_name, "hist", f_type]) + ".nc"),
                                                     "w")
                            fh_out[f_type].createDimension("lat", len(lats))
                            fh_out[f_type].createDimension("lon", len(lons))
                            outVar = fh_out[f_type].createVariable('lat', 'f4', ('lat'))
                            outVar.setncatts({"units": "degree_north"})
                            outVar[:] = lats[:]
                            outVar = fh_out[f_type].createVariable('lon', 'f4', ('lon'))
                            outVar.setncatts({"units": "degree_east"})
                            outVar[:] = lons[:]
                        first_flag = False

                    vprint(" ".join(map(str, ["Found", d, var, "train", np.bitwise_not(mask_or["train"]).sum(),
                                              "test", np.bitwise_not(mask_or["test"]).sum()])), verbose)

                    for f_type in ["train", "test"]:
                        r2 = round_sigfigs(r2_score(ma.array(fh_in[f_type].variables[var][:],
                                                             mask=mask_or[f_type]).compressed(),
                                                    ma.array(varin[i_lat_start: i_lat_end + 1,
                                                             i_lon_start: i_lon_end + 1],
                                                             mask=mask_or[f_type]).compressed()), 3)
                        vprint(" ".join(map(str, [f_type, r2])), verbose)
                        return_r2[var][f_type].append(r2)
                    vprint("-----------------------------", verbose)
                    if output:
                        doy_diff = str(abs((datetime.strptime(doy, "%Y%m%d") - datetime.strptime(d, "%Y%m%d")).days))
                        for f_type in ["train", "test"]:
                            varin = fh_comp.variables[var]
                            outVar = fh_out[f_type].createVariable(var + "_" + doy_diff, varin.datatype,
                                                                   varin.dimensions)
                            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                            outVar[:] = ma.array(varin[i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1],
                                                 mask=mask_or[f_type])
                else:
                    for f_type in ["train", "test"]:
                        r2 = round_sigfigs(r2_score(ma.array(fh_in[f_type].variables[var][:],
                                                             mask=mask_or[f_type]).compressed(),
                                                    ma.array(varin[i_lat_start: i_lat_end + 1,
                                                             i_lon_start: i_lon_end + 1],
                                                             mask=mask_or[f_type]).compressed()), 3)
                        abandoned_r2[var][f_type].append(r2)
    vprint("====================================================", verbose)
    return return_r2, abandoned_r2




# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from data_preprocessing.utils import generate_doy
from data_preprocessing.utils import get_out_path
from data_preprocessing.utils import generate_most_recent_doys

import os
from netCDF4 import Dataset
import numpy.ma as ma
from operator import or_
from datetime import date, timedelta
import numpy as np


def generate_rb_hist_average_time_window(f_name, n, doy_start, doy_end):
    fh_in = Dataset(os.path.join("Data", "Sentinel", f_name + ".nc"), "r")
    out_path = get_out_path(os.path.join("Data", "Sentinel", "usa_rb_hist_average_" + str(n)))

    init_doy, final_doy = f_name.split("_")[1], f_name.split("_")[2]
    init_doy = date(*map(int, [init_doy[:4], init_doy[4:6], init_doy[6:]]))
    final_doy = date(*map(int, [final_doy[:4], final_doy[4:6], final_doy[6:]]))

    doy_s = date(*map(int, [doy_start[:4], doy_start[4:6], doy_start[6:]]))
    doy_e = date(*map(int, [doy_end[:4], doy_end[4:6], doy_end[6:]]))
    assert((doy_s-init_doy).days >= n)
    assert((final_doy-doy_e).days >= 0)

    i_doy = (doy_s - init_doy).days
    for doy in generate_doy(doy_start, doy_end, ""):
        fh_out = Dataset(os.path.join(out_path, doy + ".nc"), "w")

        for name, dim in fh_in.dimensions.items():
            if name != "time":
                fh_out.createDimension(name, len(dim))

        datatype = None
        s_doy = 0
        hist_3km, hist_9km = {}, {}
        for v_name, varin in fh_in.variables.items():
            if v_name == 'lat' or v_name == 'lon':
                outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                outVar[:] = varin[:]
            elif v_name != "time":
                if "9km" not in v_name:
                    outVar = fh_out.createVariable(v_name + "_hist_mean_" + str(n), varin.datatype, ("lat", "lon",))
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    s_doy = i_doy - n
                    hist_3km[v_name[:9]] = ma.mean(varin[s_doy:i_doy, :, :], axis=0)
                    outVar[:] = hist_3km[v_name[:9]]
                    outVar = fh_out.createVariable(v_name + "_hist_std_" + str(n), varin.datatype, ("lat", "lon",))
                    outVar[:] = ma.std(varin[s_doy:i_doy, :, :], axis=0)
                else:
                    datatype = varin.datatype
                    outVar = fh_out.createVariable(v_name + "_hist_" + str(n), varin.datatype, ("lat", "lon",))
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    hist_9km[v_name[:9]] = ma.mean(varin[s_doy:i_doy, :, :], axis=0)
                    outVar[:] = hist_9km[v_name[:9]]

        print(s_doy, i_doy)
        for v_name in ["sigma0_vh", "sigma0_vv"]:
            outVar = fh_out.createVariable(v_name + "_diff_hist_" + str(n), datatype, ("lat", "lon"))
            outVar[:] = hist_3km[v_name] - hist_9km[v_name]

        i_doy += 1
        fh_out.close()

    fh_in.close()


def generate_rb_hist_average_time_window_by_doy(doys, n, var_list):
    out_path = get_out_path(os.path.join("Data", "Sentinel", "usa_rb_hist_average_" + str(n)))
    in_path = os.path.join("Data", "Sentinel", "usa_9km_all")

    for doy in doys:
        fh_in = Dataset(os.path.join(in_path, doy + ".nc"), "r")
        fh_out = Dataset(os.path.join(out_path, doy + ".nc"), "w")

        hist_vars = {}
        datatype = None
        n_lat, n_lon = len(fh_in.variables["lat"][:]), len(fh_in.variables["lon"][:])
        for name, dim in fh_in.dimensions.items():
            fh_out.createDimension(name, len(dim))

        for v_name, varin in fh_in.variables.items():
            if v_name in var_list:
                if "9km" not in v_name:
                    outVar = fh_out.createVariable(v_name + "_hist_mean_" + str(n), varin.datatype,
                                                   ("lat", "lon",))
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    fh_out.createVariable(v_name + "_hist_std_" + str(n), varin.datatype, ("lat", "lon",))
                    hist_vars[v_name] = ma.empty([30, n_lat, n_lon])
                else:
                    datatype = varin.datatype
                    outVar = fh_out.createVariable(v_name + "_hist_" + str(n), varin.datatype, ("lat", "lon",))
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    hist_vars[v_name] = ma.empty([30, n_lat, n_lon])
            elif v_name == "lat" or v_name == "lon":
                outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                outVar[:] = varin[:]

        i_hist = 0
        for hist_doy in generate_most_recent_doys(doy, n, ""):
            fh_hist = Dataset(os.path.join(in_path, hist_doy + ".nc"), "r")
            for v_name, varin in fh_hist.variables.items():
                if v_name in var_list:
                    hist_vars[v_name][i_hist] = varin[:]
            i_hist += 1
            fh_hist.close()

        for v_name in hist_vars.keys():
            if "9km" in v_name:
                fh_out.variables[v_name+"_hist_"+str(n)][:] = ma.mean(hist_vars[v_name], axis=0)
            else:
                fh_out.variables[v_name+"_hist_mean_"+str(n)][:] = ma.mean(hist_vars[v_name], axis=0)
                fh_out.variables[v_name + "_hist_std_" + str(n)][:] = ma.std(hist_vars[v_name], axis=0)

        for v_name in ["sigma0_vh", "sigma0_vv"]:
            outVar = fh_out.createVariable(v_name + "_diff_hist_" + str(n), datatype, ("lat", "lon"))
            outVar[:] = ma.mean(hist_vars[v_name+"_aggregated"], axis=0) - \
                        ma.mean(hist_vars[v_name+"_aggregated_9km_mean"], axis=0)
        fh_in.close()
        fh_out.close()


def generate_rb_hist_n(f_name, n, doy_start, doy_end):
    fh_in = Dataset(os.path.join("Data", "Sentinel", f_name + ".nc"), "r")
    out_path = get_out_path(os.path.join("Data", "Sentinel", "usa_rb_hist_" + str(n)))

    init_doy, final_doy = f_name.split("_")[1], f_name.split("_")[2]
    init_doy = date(*map(int, [init_doy[:4], init_doy[4:6], init_doy[6:]]))
    final_doy = date(*map(int, [final_doy[:4], final_doy[4:6], final_doy[6:]]))

    doy_s = date(*map(int, [doy_start[:4], doy_start[4:6], doy_start[6:]]))
    doy_e = date(*map(int, [doy_end[:4], doy_end[4:6], doy_end[6:]]))
    assert((doy_s-init_doy).days >= n)
    assert((final_doy-doy_e).days >= 0)

    i_doy = (doy_s - init_doy).days
    for doy in generate_doy(doy_start, doy_end, ""):
        fh_out = Dataset(os.path.join(out_path, doy + ".nc"), "w")

        for name, dim in fh_in.dimensions.items():
            if name != "time":
                fh_out.createDimension(name, len(dim))

        for v_name, varin in fh_in.variables.items():
            if v_name == 'lat' or v_name == 'lon':
                outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                outVar[:] = varin[:]
            elif v_name != "time":
                outVar = fh_out.createVariable(v_name + "_hist_mean_" + str(n), varin.datatype, ("lat", "lon",))
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                s_doy = i_doy - n
                print(s_doy, i_doy)
                outVar[:] = ma.mean(varin[s_doy:i_doy, :, :], axis=0)
                outVar = fh_out.createVariable(v_name + "_hist_std_" + str(n), varin.datatype, ("lat", "lon",))
                outVar[:] = ma.std(varin[s_doy:i_doy, :, :], axis=0)
        i_doy += 1

        fh_out.close()

    fh_in.close()

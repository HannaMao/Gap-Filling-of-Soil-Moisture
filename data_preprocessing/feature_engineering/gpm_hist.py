# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from data_preprocessing.utils import generate_doy
from data_preprocessing.utils import get_out_path
from data_preprocessing.utils import generate_most_recent_doys

import os
from netCDF4 import Dataset


def generate_gpm_hist(f_name):
    fh_in = Dataset(os.path.join("Data", "GPM", f_name + ".nc"), "r")
    out_path = get_out_path(os.path.join("Data", "GPM", "hist_added"))

    doy_start, doy_end = f_name.split("_")[1], f_name.split("_")[2]
    n_days = [1, 2, 3]

    i_doy = 0
    for doy in generate_doy(doy_start, doy_end, ""):
        if i_doy < 3:
            i_doy += 1
            continue
        else:
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
                    outVar = fh_out.createVariable(v_name, varin.datatype, ("lat", "lon",))
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    outVar[:] = varin[i_doy, :, :]
                    for n in n_days:
                        outVar = fh_out.createVariable(v_name + "_hist" + str(n), varin.datatype, ("lat", "lon",))
                        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                        outVar[:] = varin[i_doy-n, :, :]
            i_doy += 1
            fh_out.close()

    fh_in.close()


def generate_gpm_hist_by_doy(doys):
    out_path = get_out_path(os.path.join("Data", "GPM", "hist_added"))
    for doy in doys:
        fh_in = Dataset(os.path.join("Data", "GPM", "3km", doy + ".nc"), "r")
        fh_out = Dataset(os.path.join(out_path, doy + ".nc"), "w")

        for name, dim in fh_in.dimensions.items():
            fh_out.createDimension(name, len(dim))

        for v_name, varin in fh_in.variables.items():
            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[:]

        n = 1
        for hist_doy in generate_most_recent_doys(doy, 3, ""):
            fh_hist = Dataset(os.path.join("Data", "GPM", "3km", hist_doy + ".nc"), "r")
            for v_name, varin in fh_hist.variables.items():
                if v_name != "lat" and v_name != "lon":
                    outVar = fh_out.createVariable(v_name + "_hist" + str(n), varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    outVar[:] = varin[:]
            fh_hist.close()
            n += 1

        fh_out.close()
        fh_in.close()


# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..utils import select_area, get_lat_lon, match_lat_lon, get_out_path, generate_doy

import os
import numpy.ma as ma
import numpy as np
from netCDF4 import Dataset


def merge_rb_rb_hist(rb_hist_folder, s_doy, e_doy):
    # check the coverage of rb_hist compared with current rb
    rb_path = os.path.join("Data", "Sentinel", "usa_db")
    rb_hist_path = os.path.join("Data", "Sentinel", rb_hist_folder)
    out_path = get_out_path(os.path.join("Data", "Sentinel", "check_" + rb_hist_folder + "_coverage"))

    for doy in generate_doy(s_doy, e_doy, ""):
        print(doy)
        fh_dic = dict()
        fh_dic["rb"] = Dataset(os.path.join(rb_path, doy + ".nc"), "r")
        fh_dic["rb_hist"] = Dataset(os.path.join(rb_hist_path, doy + ".nc"), "r")
        fh_out = Dataset(os.path.join(out_path, doy + ".nc"), "w")

        for name, dim in fh_dic["rb"].dimensions.items():
            fh_out.createDimension(name, len(dim))

        for v_name, varin in fh_dic["rb"].variables.items():
            if v_name == 'lat' or v_name == 'lon':
                outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                outVar[:] = varin[:]

        ma_dic = {}
        for fName in fh_dic:
            for v_name, varin in fh_dic[fName].variables.items():
                if v_name != "lat" and v_name != "lon":
                    outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    outVar[:] = varin[:]
                    ma_dic[v_name] = ma.getmaskarray(varin[:])

        daily_mask = np.logical_or.reduce(list(ma_dic.values()))

        n_before = fh_out.variables["sigma0_vh_aggregated"][:].count()
        print("Before mask, number of valid grids:", n_before)

        for var in fh_out.variables:
            if var != "lat" and var != "lon":
                fh_out.variables[var][:] = ma.array(fh_out.variables[var][:], mask=daily_mask)

        n_after = fh_out.variables["sigma0_vh_aggregated"][:].count()
        print("After mask, number of valid grids:", n_after)
        print("Percentage:", n_after / n_before)
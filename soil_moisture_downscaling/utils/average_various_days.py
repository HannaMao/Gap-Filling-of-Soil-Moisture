# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from netCDF4 import Dataset
from collections import defaultdict
import numpy.ma as ma


def average_various_days(in_file_list, out_file, selected_vars=None):
    fh_out = Dataset(out_file, "w")

    var_lis = defaultdict(list)
    first = True
    for in_file in in_file_list:
        fh_in = Dataset(in_file, "r")

        for v_name, varin in fh_in.variables.items():
            if selected_vars is None or v_name in selected_vars:
                var_lis[v_name].append(fh_in.variables[v_name][:])

        if first:
            for name, dim in fh_in.dimensions.items():
                fh_out.createDimension(name, len(dim))
            for v_name, varin in fh_in.variables.items():
                if selected_vars is None or v_name in selected_vars or v_name in ["lat", "lon"]:
                    outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    outVar[:] = varin[:]
            first = False
        fh_in.close()

    for var in fh_out.variables:
        if var != "lat" and var != "lon":
            fh_out.variables[var][:] = ma.array(var_lis[var]).mean(axis=0)
    fh_out.close()


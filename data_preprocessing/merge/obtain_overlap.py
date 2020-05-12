# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..preprocess import modis_lst_fill_missing_by_most_recent_values
from ..utils import select_area, get_lat_lon, match_lat_lon, get_out_path

import os
import numpy.ma as ma
import numpy as np
from netCDF4 import Dataset


def obtain_overlap(in_files_lis, out_file, variable):
    fh_out = Dataset(out_file, "w")

    ma_lis = []
    for f_index, in_file in enumerate(in_files_lis):
        fh_in = Dataset(in_file, "r")
        ma_lis.append(ma.getmaskarray(fh_in.variables[variable][:]))

        if f_index == len(in_files_lis) - 1:
            for name, dim in fh_in.dimensions.items():
                fh_out.createDimension(name, len(dim))
            overlap_mask = np.logical_or.reduce(ma_lis)
            for v_name, varin in fh_in.variables.items():
                if v_name == 'lat' or v_name == 'lon':
                    outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    outVar[:] = varin[:]
                else:
                    outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    outVar[:] = ma.array(varin[:], mask=overlap_mask)

        fh_in.close()
    fh_out.close()















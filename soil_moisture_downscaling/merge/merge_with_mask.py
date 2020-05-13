# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import numpy.ma as ma
from netCDF4 import Dataset


def merge_with_mask(in_file, out_file, mask):
    fh_out = Dataset(out_file, "w")
    fh_in = Dataset(in_file, "r")

    for name, dim in fh_in.dimensions.items():
        fh_out.createDimension(name, len(dim))

    for v_name, varin in fh_in.variables.items():
        if v_name == 'lat' or v_name == 'lon':
            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[:]
        else:
            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = ma.array(varin[:], mask=mask)

    fh_out.close()

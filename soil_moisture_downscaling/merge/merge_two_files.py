# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from netCDF4 import Dataset
import numpy.ma as ma


def merge_two_files(file1, v1, file2, v2, out_file):
    fh_1 = Dataset(file1, "r")
    fh_2 = Dataset(file2, "r")

    fh_out = Dataset(out_file, "w")

    for name, dim in fh_1.dimensions.items():
        fh_out.createDimension(name, len(dim))

    for v_name, varin in fh_1.variables.items():
        if v_name in ['lat', 'lon']:
            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[:]
        elif v_name == v1:
            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = ma.array((fh_1.variables[v1][:], fh_2.variables[v2][:])).mean(axis=0)

    fh_1.close()
    fh_2.close()
    fh_out.close()

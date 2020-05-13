# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause


from netCDF4 import Dataset
from collections import defaultdict
import numpy.ma as ma


def merge_predict_with_orig(predict_file, origi_file, out_file, predicted_var):
    fh_out = Dataset(out_file, "w")

    var_lis = []
    fh_predict = Dataset(predict_file, "r")
    var_lis.append(fh_predict.variables[predicted_var + "_predicted"][:])
    fh_origi = Dataset(origi_file, "r")
    var_lis.append(fh_origi.variables[predicted_var][:])

    for name, dim in fh_origi.dimensions.items():
        fh_out.createDimension(name, len(dim))
    for v_name, varin in fh_origi.variables.items():
        if v_name == predicted_var or v_name in ["lat", "lon"]:
            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[:]

    for var in fh_out.variables:
        if var != "lat" and var != "lon":
            fh_out.variables[var][:] = ma.array(var_lis).mean(axis=0)

    fh_predict.close()
    fh_origi.close()
    fh_out.close()


# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..utils import get_out_path

import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma


def merge_prediction(base_path, out_path, var_names, data_params, feature_index, n_cells):
    tuning_case, area_folder, doy_folder, model = data_params
    dims = ["time", "lat", "lon"] if "time" in doy_folder else ["lat", "lon"]

    for n_cell in n_cells:
        for f_type in ["train", "test"]:
            fh = Dataset(os.path.join(base_path, str(n_cell * 3) + "km_" + f_type + ".nc"), "r")
            fh_out = Dataset(os.path.join(out_path, str(n_cell * 3) + "km_" + f_type + ".nc"), "w")

            for dim in fh.dimensions:
                fh_out.createDimension(dim, fh.dimensions[dim].size)

            get_mask = False
            for v_name, varin in fh.variables.items():
                outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                outVar[:] = varin[:]
                if not get_mask and v_name not in dims:
                    base_mask = ma.getmaskarray(varin[:])
                    get_mask = True

            for var_name in var_names:
                data_path = os.path.join("STTL_Spatial_Comparison_Experiments", tuning_case, area_folder, doy_folder,
                                         "_".join([var_name, model]), feature_index)
                fh_prediction = Dataset(os.path.join(
                    data_path, str(n_cell * 3) + "km_prediction_" + f_type + ".nc"), "r")

                varin = fh_prediction[var_name + "_predicted"]
                assert np.array_equal(base_mask, ma.getmaskarray(varin[:]))
                outVar = fh_out.createVariable(var_name + "_predicted", varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                outVar[:] = varin[:]

                fh_prediction.close()
            fh.close()
            fh_out.close()
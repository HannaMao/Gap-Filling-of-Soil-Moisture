# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..utils import get_out_path

import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma


def merge_prediction(big_folder, experiment_folder, base_path, out_path, var_names, data_params, feature_index, fname,
                     predicted_name, real_gap=False):
    tuning_case, area_folder, doy_folder, model = data_params
    dims = ["time", "lat", "lon"] if "time" in doy_folder else ["lat", "lon"]

    fh = Dataset(os.path.join(base_path, fname), "r")
    get_out_path(out_path)
    fh_out = Dataset(os.path.join(out_path, fname), "w")

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
        data_path = os.path.join(big_folder, experiment_folder, tuning_case, area_folder,
                                 doy_folder, "_".join([var_name, model]), feature_index)
        if real_gap:
            fh_prediction = Dataset(os.path.join(data_path, "train_temporal", predicted_name), "r")
        else:
            fh_prediction = Dataset(os.path.join(data_path, predicted_name), "r")

        varin = fh_prediction[var_name + "_predicted"]
        predicted_mask = ma.getmaskarray(varin[:])
        mask_diff = np.sum(predicted_mask) - np.sum(base_mask)
        # assert 0 <= mask_diff < 5
        outVar = fh_out.createVariable(var_name + "_predicted", varin.datatype, varin.dimensions)
        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
        if mask_diff > 0:
            print("WARNING: mask diff:", mask_diff)
            filled_out = varin[:]
            i, j = np.where(predicted_mask ^ base_mask)
            for ii, jj in zip(i, j):
                filled_out[ii, jj] = ma.mean(varin[:])
            outVar[:] = ma.array(filled_out, mask=base_mask)
        else:
            outVar[:] = varin[:]

        fh_prediction.close()
    fh.close()
    fh_out.close()

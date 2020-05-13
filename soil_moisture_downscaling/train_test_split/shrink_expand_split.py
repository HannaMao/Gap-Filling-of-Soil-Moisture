# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..utils import get_out_path

import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma


def get_shrink_mask_surrounded(in_mask, n_cells):
    train_mask = np.full(in_mask.shape, False)
    test_mask = np.full(in_mask.shape, False)

    for j in range(in_mask.shape[1]):
        upper_start = lower_end = -1
        for i in range(in_mask.shape[0]):
            if not in_mask[i, j]:
                upper_start = i
                break
        for i in range(in_mask.shape[0] - 1, -1, -1):
            if not in_mask[i, j]:
                lower_end = i
                break
        if upper_start != -1:
            train_mask[upper_start: (upper_start + n_cells), j] = True
        if lower_end != -1:
            train_mask[(lower_end - n_cells + 1): (lower_end + 1), j] = True

    for i in range(in_mask.shape[0]):
        left_start = right_end = -1
        for j in range(in_mask.shape[1]):
            if not in_mask[i, j]:
                left_start = j
                break
        for j in range(in_mask.shape[1] - 1, -1, -1):
            if not in_mask[i, j]:
                right_end = j
                break
        if left_start != -1:
            train_mask[i, left_start: (left_start + n_cells)] = True
        if right_end != -1:
            train_mask[i, (right_end - n_cells + 1): (right_end + 1)] = True

    for i in range(in_mask.shape[0]):
        for j in range(in_mask.shape[1]):
            if not train_mask[i, j]:
                test_mask[i, j] = True

    return train_mask, test_mask


def get_shrink_mask_vertical_strips_double_sides(in_mask, n_cells):
    train_mask = np.full(in_mask.shape, False)
    test_mask = np.full(in_mask.shape, False)

    for i in range(in_mask.shape[0]):
        i_start = i_end = -1
        for j in range(in_mask.shape[1]):
            if not in_mask[i, j]:
                i_start = j
                break
        for j in range(in_mask.shape[1] - 1, -1, -1):
            if not in_mask[i, j]:
                i_end = j
                break
        if i_start != -1 and i_end != -1:
            train_mask[i, i_start: (i_start + n_cells)] = True
            train_mask[i, (i_end + 1 - n_cells): i_end + 1] = True
            test_mask[i, (i_start + n_cells): (i_end + 1 - n_cells)] = True

    return train_mask, test_mask


def get_shrink_mask_vertical_strips_single_side(in_mask, n_cells):
    train_mask = np.full(in_mask.shape, False)
    test_mask = np.full(in_mask.shape, False)

    for i in range(in_mask.shape[0]):
        i_start = -1
        for j in range(in_mask.shape[1]):
            if not in_mask[i, j]:
                i_start = j
                break
        if i_start != -1:
            train_mask[i, i_start: (i_start + n_cells)] = True
            test_mask[i, (i_start + n_cells):] = True

    return train_mask, test_mask


def shrink_expand_split(in_path, f_name, n_cells, out_path, split_type):
    fh_in = Dataset(os.path.join(in_path, f_name + ".nc"), "r")
    fh_dic = dict()
    fh_dic["train"] = Dataset(os.path.join(out_path, str(n_cells * 3) + "km_train.nc"), "w")
    fh_dic["test"] = Dataset(os.path.join(out_path, str(n_cells * 3) + "km_test.nc"), "w")

    for dim in fh_in.dimensions:
        fh_dic["train"].createDimension(dim, fh_in.dimensions[dim].size)
        fh_dic["test"].createDimension(dim, fh_in.dimensions[dim].size)

    mask_dic = {}
    in_mask = ma.getmaskarray(fh_in.variables["soil_moisture"][:])
    if split_type == "vertical_split_double":
        mask_dic["train"], mask_dic["test"] = get_shrink_mask_vertical_strips_double_sides(in_mask, n_cells)
    elif split_type == "vertical_split_single":
        mask_dic["train"], mask_dic["test"] = get_shrink_mask_vertical_strips_single_side(in_mask, n_cells)
    elif split_type == "surrounded_split":
        mask_dic["train"], mask_dic["test"] = get_shrink_mask_surrounded(in_mask, n_cells)
    for v_name, varin in fh_in.variables.items():
        for data_type in fh_dic:
            outVar = fh_dic[data_type].createVariable(v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            if v_name == "lat" or v_name == "lon":
                outVar[:] = varin[:]
            else:
                outVar[:] = ma.array(varin[:], mask=np.logical_or(in_mask, mask_dic[data_type]))

    fh_in.close()
    fh_dic["train"].close()
    fh_dic["test"].close()



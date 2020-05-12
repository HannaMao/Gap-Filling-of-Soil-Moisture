# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from .timing import timeit, timenow
from .get_lat_lon import get_lat_lon, match_lat_lon
from .generate_doy import generate_doy, generate_nearest_doys, generate_most_recent_doys
from .select_area import select_area
from .find_index import find_index
from .logger import Logger
from .float_precision import round_sigfigs
from .generate_mask import generate_mask, generate_numpy_mask, get_variable_mask, get_two_variables_mask
from .mask_files import mask_by_mask_array, mask_by_state
from .unique_landcover_by_state import obtain_unique_landcover_by_state
from .subset_usa import subset_usa

import os
import pickle

__all__ = ["get_out_path",
           "timeit", "timenow",
           "get_lat_lon", "match_lat_lon",
           "generate_doy", "generate_nearest_doys", "generate_most_recent_doys",
           "select_area",
           "generate_mask", "generate_numpy_mask", "get_variable_mask", "get_two_variables_mask",
           "mask_by_mask_array", "mask_by_state",
           "obtain_unique_landcover_by_state",
           "subset_usa",
           "save_pkl", "load_pkl"]


def get_out_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def vprint(output, verbose):
    if verbose:
        print(output)


def save_pkl(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(f):
    with open(f, "rb") as pkl:
        return pickle.load(pkl)

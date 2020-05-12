# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import os
from netCDF4 import Dataset
import numpy.ma as ma
import numpy as np


def obtain_unique_landcover_by_state(state_index):
    fh_landcover = Dataset(os.path.join("Data", "Sentinel", "landcover_class_usa.nc"), "r")
    fh_states = Dataset(os.path.join("Data", "US_States", "usa_states.nc"), "r")
    states_array = fh_states.variables["states_flag"][:]
    mask_array = ma.getmaskarray(ma.masked_where(states_array != state_index, states_array))

    return map(int, sorted(np.unique(ma.array(fh_landcover.variables["landcover_class"][:], mask=mask_array)))[:-1])

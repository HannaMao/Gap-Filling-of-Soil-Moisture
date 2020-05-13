# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from .get_lat_lon import get_lat_lon


def select_area(lat1, lat2, lon1, lon2, reso):
    """
    left-up: lat1 lon1  left-down: lat2 lon1  right-up: lat1 lon2  right-down: lat2 lon2
    e.g.
    For Oklahoma
    :param lat1: 38
    :param lat2: 32
    :param lon1: -104
    :param lon2: -92
    """
    lats, lons = get_lat_lon(reso)

    lat_indices = lats.size - np.searchsorted(lats[::-1], [lat1, lat2], side="right")
    lon_indices = np.searchsorted(lons, [lon1, lon2])

    return lat_indices, lon_indices




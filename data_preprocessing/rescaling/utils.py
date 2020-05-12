# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause
from ..utils import get_lat_lon, select_area

import numpy as np


def get_lat_lon_bins(reso, lat1, lat2, lon1, lon2):
    lat_indices, lon_indices = select_area(lat1, lat2, lon1, lon2, reso)
    lats, lons = get_lat_lon(reso)
    assert (len(lats) != 0 and len(lons) != 0)
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]

    inter_lat = np.array([(x + y) / 2.0 for x, y in zip(lats[:-1], lats[1:])])
    inter_lon = np.array([(x + y) / 2.0 for x, y in zip(lons[:-1], lons[1:])])
    lat_bins = np.concatenate([[2 * inter_lat[0] - inter_lat[1]], inter_lat, [2 * inter_lat[-1] - inter_lat[-2]]])
    lon_bins = np.concatenate([[2 * inter_lon[0] - inter_lon[1]], inter_lon, [2 * inter_lon[-1] - inter_lon[-2]]])

    return lats, lons, lat_bins, lon_bins
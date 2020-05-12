# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from scipy.spatial import cKDTree
import numpy as np
from numpy import cos, sin, sqrt
from math import pi


def distance(lat_1, lon_1, lat_2, lon_2):
    rad_factor = pi/180.0
    lat_1 *= rad_factor
    lon_1 *= rad_factor
    lat_2 *= rad_factor
    lon_2 *= rad_factor

    clat, clon = cos(lat_1), cos(lon_1)
    slat, slon = sin(lat_1), sin(lon_1)
    delX = cos(lat_2) * cos(lon_2) - clat * clon
    delY = cos(lat_2) * sin(lon_2) - clat * slon
    delZ = sin(lat_2) - slat
    dist_sq = sqrt(delX ** 2 + delY ** 2 + delZ ** 2)
    return dist_sq


def bilinear(lat_center, lon_center, iy, ix, latvar, lonvar, target_var, mask_value=None):
    interpolated_value = 0
    sum_area = 0

    for x, y, x1, y1, i in zip(ix[::-1], iy[::-1], ix, iy, range(len(iy))):
        if target_var[y, x] is not np.ma.masked:
            area = distance(lat_center, lon_center, latvar[y1], lon_center) * \
                   distance(lat_center, lon_center, lat_center, lonvar[x1])
            interpolated_value += target_var[y, x] * area
            sum_area += area
    if sum_area != 0:
        interpolated_value = interpolated_value / sum_area
    else:
        if mask_value is not None:
            interpolated_value = mask_value

    return interpolated_value


class Kdtree_fast(object):
    def __init__(self, ncfile, latvarname, lonvarname):
        self.ncfile = ncfile
        self.latvar = self.ncfile.variables[latvarname]
        self.lonvar = self.ncfile.variables[lonvarname]

        rad_factor = pi / 180.0
        self.latvals = self.latvar[:] * rad_factor
        self.lonvals = self.lonvar[:] * rad_factor
        ny = len(self.latvals)
        nx = len(self.lonvals)
        self.latvals = np.repeat(self.latvals, nx, axis=0).reshape(ny, nx)
        self.lonvals = np.tile(self.lonvals, [ny]).reshape(ny, nx)
        self.shape = self.latvals.shape

        clat, clon = cos(self.latvals), cos(self.lonvals)
        slat, slon = sin(self.latvals), sin(self.lonvals)
        clat_clon = clat * clon
        clat_slon = clat * slon
        triples = list(zip(np.ravel(clat_clon), np.ravel(clat_slon), np.ravel(slat)))
        self.kdt = cKDTree(triples)

    def query(self, lat0, lon0):
        rad_factor = pi / 180.0
        lat0_rad = lat0 * rad_factor
        lon0_rad = lon0 * rad_factor
        clat0, clon0 = cos(lat0_rad), cos(lon0_rad)
        slat0, slon0 = sin(lat0_rad), sin(lon0_rad)
        dist_sq_min, minindex_1d = self.kdt.query(
            [clat0 * clon0, clat0 * slon0, slat0], k=4)
        iy_min, ix_min = np.unravel_index(minindex_1d, self.shape)
        return dist_sq_min, iy_min, ix_min
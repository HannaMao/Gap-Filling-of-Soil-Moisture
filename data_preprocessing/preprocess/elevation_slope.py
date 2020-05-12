# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..utils import get_lat_lon, get_out_path

import os
import numpy as np
from netCDF4 import Dataset
import numpy.ma as ma


def elevation_slope_extract():
    out_path = get_out_path(os.path.join("Data", "Elevation"))
    with open(os.path.join("n5eil01u.ecs.nsidc.org", "Elevation", "DEMSLP_M03_003.float32")) as f:
        slope_values = np.fromfile(f, '<f4').reshape([11568, 4872])

        lats, lons = get_lat_lon("M03")

        fh_out = Dataset(os.path.join(out_path, "slope.nc"), "w")
        fh_out.createDimension("lat", len(lats))
        fh_out.createDimension("lon", len(lons))

        outVar = fh_out.createVariable('lat', float, ('lat'))
        outVar.setncatts({"units": "degree_north"})
        outVar[:] = lats[:]
        outVar = fh_out.createVariable('lon', float, ('lon'))
        outVar.setncatts({"units": "degree_east"})
        outVar[:] = lons[:]

        slope_var = fh_out.createVariable('slope', float, ('lat', "lon"))
        slope_var[:] = slope_values

        fh_out.close()

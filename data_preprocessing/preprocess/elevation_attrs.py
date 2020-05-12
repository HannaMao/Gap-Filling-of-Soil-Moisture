# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import os
import numpy as np
import gdal
from netCDF4 import Dataset
import numpy.ma as ma


def elevation_attrs_convert_to_nc(attr_name):
    im = gdal.Open(os.path.join("n5eil01u.ecs.nsidc.org", "Elevation", attr_name + ".tif"))
    imarray = im.ReadAsArray()

    n_lat, n_lon = np.shape(imarray)
    b = im.GetGeoTransform()  # bbox, interval
    lons = np.arange(n_lon) * b[1] + b[0]
    lats = np.arange(n_lat) * b[5] + b[3]

    fh_out = Dataset(os.path.join("Data", "Elevation", attr_name + ".nc"), "w")
    fh_out.createDimension("lat", len(lats))
    fh_out.createDimension("lon", len(lons))

    outVar = fh_out.createVariable('lat', float, ('lat'))
    outVar.setncatts({"units": "degree_north"})
    outVar[:] = lats[:]
    outVar = fh_out.createVariable('lon', float, ('lon'))
    outVar.setncatts({"units": "degree_east"})
    outVar[:] = lons[:]

    outVar = fh_out.createVariable(attr_name, float, ("lat", "lon"))
    outVar[:] = ma.masked_equal(imarray, np.min(imarray))







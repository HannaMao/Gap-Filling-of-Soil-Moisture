# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import os
import numpy as np
import gdal
from netCDF4 import Dataset
import numpy.ma as ma


def bulk_density_convert_to_nc():
    im = gdal.Open(os.path.join("n5eil01u.ecs.nsidc.org", "Bulk_Density", "bulk_density_latlon_1km.tif"))
    imarray = im.ReadAsArray()
    n_lat, n_lon = np.shape(imarray)
    b = im.GetGeoTransform()  # bbox, interval
    lons = np.arange(n_lon) * b[1] + b[0]
    lats = np.arange(n_lat) * b[5] + b[3]

    fh_out = Dataset(os.path.join("Data", "Bulk_Density", "bulk_density_1km.nc"), "w")
    fh_out.createDimension("lat", len(lats))
    fh_out.createDimension("lon", len(lons))

    outVar = fh_out.createVariable('lat', float, ('lat'))
    outVar.setncatts({"units": "degree_north"})
    outVar[:] = lats[:]
    outVar = fh_out.createVariable('lon', float, ('lon'))
    outVar.setncatts({"units": "degree_east"})
    outVar[:] = lons[:]

    outVar = fh_out.createVariable("bulk_density", float, ("lat", "lon"))
    outVar[:] = ma.masked_less(imarray, 0)







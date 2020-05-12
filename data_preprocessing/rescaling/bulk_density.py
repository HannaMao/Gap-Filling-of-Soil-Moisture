# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from .utils import get_lat_lon_bins

import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma


def bulk_density_upsample(lat1, lat2, lon1, lon2, reso, area_name):
    fh_in = Dataset(os.path.join("Data", "Bulk_Density", "bulk_density_1km.nc"), "r")
    fh_out = Dataset(os.path.join("Data", "Bulk_Density", "bulk_density_" + reso + "_" + area_name + ".nc"), "w")

    n_dim = str(int(reso[:-2])).zfill(2)
    lats, lons, lat_bins, lon_bins = get_lat_lon_bins("M" + n_dim, lat1, lat2, lon1, lon2)

    bd_lats = fh_in.variables['lat']
    bd_lats_value = bd_lats[:][::-1]
    bd_lons = fh_in.variables['lon']
    bd_lons_value = bd_lons[:]
    bd_var = fh_in.variables['bulk_density'][:]
    bd_resampled = np.full((len(lats), len(lons)), -9999.0)

    for id_lats in range(len(lats)):
        for id_lons in range(len(lons)):
            lats_index = np.searchsorted(bd_lats_value, [lat_bins[id_lats + 1], lat_bins[id_lats]])
            lons_index = np.searchsorted(bd_lons_value, [lon_bins[id_lons], lon_bins[id_lons + 1]])
            if lats_index[0] != lats_index[1] and lons_index[0] != lons_index[1]:
                bd_selected = bd_var[np.array(range(-lats_index[1], -lats_index[0]))[:, None],
                                       np.array(range(lons_index[0], lons_index[1]))]
                avg = ma.mean(bd_selected)
                bd_resampled[id_lats, id_lons] = (avg if avg is not ma.masked else -9999.0)

    bd_resampled = ma.masked_equal(bd_resampled, -9999.0)

    fh_out.createDimension('lat', len(lats))
    fh_out.createDimension('lon', len(lons))

    outVar = fh_out.createVariable('lat', 'f4', ('lat',))
    outVar.setncatts({k: bd_lats.getncattr(k) for k in bd_lats.ncattrs()})
    outVar[:] = lats[:]

    outVar = fh_out.createVariable('lon', 'f4', ('lon',))
    outVar.setncatts({k: bd_lons.getncattr(k) for k in bd_lons.ncattrs()})
    outVar[:] = lons[:]

    outVar = fh_out.createVariable('bulk_density', 'f4', ('lat', 'lon',))
    outVar.setncatts({'_FillValue': np.array([-9999.0]).astype('f')})
    outVar[:] = bd_resampled[:]

    fh_in.close()
    fh_out.close()

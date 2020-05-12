# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from .utils import get_lat_lon_bins
from ..utils import get_out_path

import os
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset


def us_states_upsample():
    in_path = os.path.join("Data", "US_States", "states", "90m")
    out_path = get_out_path(os.path.join("Data", "US_States", "states", "3km"))

    lats, lons, lat_bins, lon_bins = get_lat_lon_bins("M03", 50, 24, -125, -66)
    for nc_file in os.listdir(in_path):
        if nc_file.endswith('.nc'):
            print(nc_file)
            fh = Dataset(os.path.join(in_path, nc_file), 'r')
            fh_out = Dataset(os.path.join(out_path, nc_file), 'w')

            ele_lats = fh.variables['lat']
            ele_lats_value = ele_lats[:][::-1]
            ele_lons = fh.variables['lon']
            ele_lons_value = ele_lons[:]
            ele_var = fh.variables['Band1'][0, :, :]
            ele_resampled = np.zeros((len(lats), len(lons)))

            for id_lats in range(len(lats)):
                for id_lons in range(len(lons)):
                    lats_index = np.searchsorted(ele_lats_value, [lat_bins[id_lats + 1], lat_bins[id_lats]])
                    lons_index = np.searchsorted(ele_lons_value, [lon_bins[id_lons], lon_bins[id_lons + 1]])
                    if lats_index[0] != lats_index[1] and lons_index[0] != lons_index[1]:
                        ele_selected = ele_var[np.array(range(-lats_index[1], -lats_index[0]))[:, None],
                                               np.array(range(lons_index[0], lons_index[1]))]
                        avg = ma.mean(ele_selected)
                        ele_resampled[id_lats, id_lons] = (avg if avg is not ma.masked else 0)
            ele_resampled = ma.masked_equal(ele_resampled, 0.0)

            fh_out.createDimension('lat', len(lats))
            fh_out.createDimension('lon', len(lons))

            outVar = fh_out.createVariable('lat', 'f4', ('lat',))
            outVar.setncatts({k: ele_lats.getncattr(k) for k in ele_lats.ncattrs()})
            outVar[:] = lats[:]

            outVar = fh_out.createVariable('lon', 'f4', ('lon',))
            outVar.setncatts({k: ele_lons.getncattr(k) for k in ele_lons.ncattrs()})
            outVar[:] = lons[:]

            outVar = fh_out.createVariable('elevation', 'f4', ('lat', 'lon',))
            outVar.setncatts({'units': "m"})
            outVar.setncatts({'long_name': "USGS_NED Elevation value"})
            outVar.setncatts({'_FillValue': np.array([0]).astype('f')})
            outVar[:] = ele_resampled[:]

            fh.close()
            fh_out.close()
# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from .utils import get_lat_lon_bins

import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma


def soil_fraction_upsample(lat1, lat2, lon1, lon2, reso, area_name):
    fh_in = Dataset(os.path.join("n5eil01u.ecs.nsidc.org", "Soil_Fraction", "soil_fraction_usa.nc"), "r")
    fh_out = Dataset(os.path.join("Data", "Soil_Fraction", "soil_fraction_" + reso + "_" + area_name + ".nc"), "w")

    n_dim = str(int(reso[:-2])).zfill(2)
    lats, lons, lat_bins, lon_bins = get_lat_lon_bins("M" + n_dim, lat1, lat2, lon1, lon2)

    sf_lats = fh_in.variables['lat']
    sf_lats_value = sf_lats[:][::-1]
    sf_lons = fh_in.variables['lon']
    sf_lons_value = sf_lons[:]

    fh_out.createDimension('lat', len(lats))
    fh_out.createDimension('lon', len(lons))

    outVar = fh_out.createVariable('lat', 'f4', 'lat')
    outVar.setncatts({k: sf_lats.getncattr(k) for k in sf_lats.ncattrs()})
    outVar[:] = lats[:]

    outVar = fh_out.createVariable('lon', 'f4', 'lon')
    outVar.setncatts({k: sf_lons.getncattr(k) for k in sf_lons.ncattrs()})
    outVar[:] = lons[:]

    dic_var = {}
    for var in ["sand", "clay"]:
        dic_var[var] = fh_in.variables[var][:]
        dic_var[var + "_resampled"] = np.full((len(lats), len(lons)), -9999.0)

        for id_lats in range(len(lats)):
            for id_lons in range(len(lons)):
                lats_index = np.searchsorted(sf_lats_value, [lat_bins[id_lats + 1], lat_bins[id_lats]])
                lons_index = np.searchsorted(sf_lons_value, [lon_bins[id_lons], lon_bins[id_lons + 1]])
                if lats_index[0] != lats_index[1] and lons_index[0] != lons_index[1]:
                    selected = dic_var[var][np.array(range(-lats_index[1], -lats_index[0]))[:, None],
                                               np.array(range(lons_index[0], lons_index[1]))]
                    avg = ma.mean(selected)
                    dic_var[var + "_resampled"][id_lats, id_lons] = (avg if avg is not ma.masked else -9999.0)

        dic_var[var + "_resampled"] = ma.masked_equal(dic_var[var + "_resampled"], -9999.0)

        outVar = fh_out.createVariable(var, 'f4', ('lat', 'lon',))
        outVar.setncatts({'_FillValue': np.array([-9999.0]).astype('f')})
        outVar[:] = dic_var[var + "_resampled"][:]

    fh_in.close()
    fh_out.close()

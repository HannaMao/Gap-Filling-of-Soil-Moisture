# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from .utils import get_lat_lon_bins
from ..utils import get_out_path

import os
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
from datetime import datetime


def modis_lai_upsample(doy_start, doy_end):
    in_path = os.path.join("Data", "MCD15A3H", "500m")
    out_path = get_out_path(os.path.join("Data", "MCD15A3H", "3km"))

    date_start = datetime.strptime(doy_start, "%Y%m%d").date()
    date_end = datetime.strptime(doy_end, "%Y%m%d").date()

    lats, lons, lat_bins, lon_bins = get_lat_lon_bins("M03", 50, 24, -125, -66)
    fill_value = {'Lai_500m': 11.0, 'LaiStdDev_500m': 11.0, "Fpar_500m": 1.1, "FparStdDev_500m": 1.1}
    mask_value = {'Lai_500m': 10.0, 'LaiStdDev_500m': 10.0, "Fpar_500m": 1.0, "FparStdDev_500m": 1.0}
    kept_vars = ['Lai_500m', 'Fpar_500m']

    for nc_file in os.listdir(in_path):
        if nc_file.endswith('.nc'):
            nc_date = datetime.strptime(nc_file[:-3], "%Y%m%d").date()
            if date_start <= nc_date <= date_end:
                print(nc_file)
                fh = Dataset(os.path.join(in_path, nc_file), 'r')
                fh_out = Dataset(os.path.join(out_path, nc_file), 'w')

                dic_var = {}
                for var in ['lat', 'lon']:
                    dic_var[var] = fh.variables[var]
                dic_var['lat_value'] = dic_var['lat'][::-1]
                dic_var['lon_value'] = dic_var['lon'][:]

                fh_out.createDimension('lat', len(lats))
                fh_out.createDimension('lon', len(lons))

                for var in ['lat', 'lon']:
                    outVar = fh_out.createVariable(var, 'f4', (var,))
                    outVar.setncatts({k: dic_var[var].getncattr(k) for k in dic_var[var].ncattrs()})
                    outVar[:] = lats if var == "lat" else lons

                vege, vege_value, vege_resampled = ({} for _ in range(3))
                for var in kept_vars:
                    vege[var] = fh.variables[var]
                    vege_value[var] = vege[var][:]
                    vege_resampled[var] = np.full((len(lats), len(lons)), fill_value[var])

                for id_lats in range(len(lats)):
                    for id_lons in range(len(lons)):
                        lats_index = np.searchsorted(dic_var['lat_value'],
                                                     [lat_bins[id_lats + 1], lat_bins[id_lats]])
                        lons_index = np.searchsorted(dic_var['lon_value'],
                                                     [lon_bins[id_lons], lon_bins[id_lons + 1]])
                        if lats_index[0] != lats_index[1] and lons_index[0] != lons_index[1]:
                            for var in kept_vars:
                                selected = vege_value[var][np.array(range(-lats_index[1], -lats_index[0]))[:, None],
                                                           np.array(range(lons_index[0], lons_index[1]))]
                                avg = ma.mean(selected)
                                vege_resampled[var][id_lats, id_lons] = (avg if avg is not ma.masked else fill_value[var])

                for var in kept_vars:
                    outVar = fh_out.createVariable(var.split("_")[0], 'f4', ('lat', 'lon',))
                    for attr in vege[var].ncattrs():
                        if attr != '_FillValue' and attr != 'valid_range':
                            outVar.setncatts({attr: vege[var].getncattr(attr)})
                    outVar.setncatts({'_FillValue': np.array([-1]).astype('f')})
                    outVar.setncatts({'valide_range': np.array((0, 100)).astype('f')})
                    outVar[:] = vege_resampled[var][:]
                    outVar[:] = ma.masked_greater(outVar, mask_value[var])

                fh.close()
                fh_out.close()

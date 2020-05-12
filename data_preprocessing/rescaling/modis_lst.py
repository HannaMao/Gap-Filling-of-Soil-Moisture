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


def modis_lst_upsample(doy_start, doy_end):
    in_path = os.path.join("Data", "MOD11A1", "500m")
    out_path = get_out_path(os.path.join("Data", "MOD11A1", "3km"))

    date_start = datetime.strptime(doy_start, "%Y%m%d").date()
    date_end = datetime.strptime(doy_end, "%Y%m%d").date()

    lats, lons, lat_bins, lon_bins = get_lat_lon_bins("M03", 50, 24, -125, -66)

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

                for var in ['LST_Day_1km', 'LST_Night_1km']:
                    fill_value = 0.0
                    dic_var[var] = fh.variables[var]
                    dic_var[var + '_value'] = dic_var[var][:]
                    dic_var[var + '_resampled'] = np.full((len(lats), len(lons)), fill_value)
                    for id_lats in range(len(lats)):
                        for id_lons in range(len(lons)):
                            lats_index = np.searchsorted(dic_var['lat_value'],
                                                         [lat_bins[id_lats + 1], lat_bins[id_lats]])
                            lons_index = np.searchsorted(dic_var['lon_value'],
                                                         [lon_bins[id_lons], lon_bins[id_lons + 1]])
                            if lats_index[0] != lats_index[1] and lons_index[0] != lons_index[1]:
                                selected = dic_var[var + '_value'][np.array(range(-lats_index[1], -lats_index[0]))[:, None],
                                                                   np.array(range(lons_index[0], lons_index[1]))]
                                avg = ma.mean(selected)
                                dic_var[var + '_resampled'][id_lats, id_lons] = (avg if avg is not ma.masked else fill_value)

                    v_name = "_".join(var.split("_")[:-1]) if var in ["LST_Day_1km", "LST_Night_1km"] else var
                    outVar = fh_out.createVariable(v_name, "f4", ('lat', 'lon',))
                    outVar.setncatts({'units': 'K'})
                    outVar.setncatts({'_FillValue': np.array([0]).astype('f4')})
                    outVar.setncatts({'valid_min': np.array([7500]).astype('f4')})
                    outVar.setncatts({'valid_max': np.array([65535]).astype('f4')})
                    outVar.setncatts({k: dic_var[var].getncattr(k) for k in dic_var[var].ncattrs()
                                      if k not in ['_FillValue', 'valid_min', 'valid_max']})
                    outVar[:] = dic_var[var + '_resampled'][:]
                    # outVar[:] = ma.masked_less(outVar, 150)

                fh.close()
                fh_out.close()


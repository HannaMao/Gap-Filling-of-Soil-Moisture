# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import os
import datetime
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset

from ..utils import get_out_path, generate_most_recent_doys


def modis_lst_extract(dates_folder, nc_file):
    out_path = get_out_path(os.path.join("Data", "MOD11A1", "500m"))

    fh_in = Dataset(os.path.join("n5eil01u.ecs.nsidc.org", "MOD11A1", dates_folder, nc_file + ".nc"), 'r')

    for index, n_days in enumerate(fh_in.variables['time'][:]):
        date = (datetime.datetime(2000, 1, 1, 0, 0) + datetime.timedelta(int(n_days))).strftime('%Y%m%d')
        print(date)
        fh_out = Dataset(os.path.join(out_path, date + '.nc'), 'w')

        for name, dim in fh_in.dimensions.items():
            if name != 'time':
                fh_out.createDimension(name, len(dim) if not dim.isunlimited() else None)

        ignore_features = ['time', 'crs', 'Clear_day_cov', 'Clear_night_cov', 'Day_view_angl', 'Day_view_time',
                           'Night_view_angl', 'Night_view_time', 'Emis_31', 'Emis_32', "QC_Day", "QC_Night"]
        for v_name, varin in fh_in.variables.items():
            if v_name not in ignore_features:
                dimensions = varin.dimensions if v_name in ['lat', 'lon'] else ('lat', 'lon')
                outVar = fh_out.createVariable(v_name, varin.datatype, dimensions)
                if v_name == "lat":
                    outVar.setncatts({"units": "degree_north"})
                    outVar[:] = varin[:]
                elif v_name == "lon":
                    outVar.setncatts({"units": "degree_east"})
                    outVar[:] = varin[:]
                else:
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    outVar[:] = varin[index, :, :]

        fh_out.close()
    fh_in.close()


def modis_lst_fill_missing_by_most_recent_values(doy, area, n_days):
    fh_in = Dataset(os.path.join("Data", "MOD11A1", "3km", doy + ".nc"), "r")

    n_lat = fh_in.dimensions["lat"].size
    n_lon = fh_in.dimensions["lon"].size
    lst_day_ref = np.ma.empty((n_days, n_lat, n_lon))
    i = 0
    for i_doy in generate_most_recent_doys(doy, n_days, ""):
        print(i_doy)
        fh = Dataset(os.path.join("Data", "MOD11A1", "3km", i_doy + ".nc"), "r")
        lst_day_ref[i, :, :] = fh.variables["LST_Day"][:]
        i += 1
        fh.close()
    lst_day_ref_mask = ma.getmaskarray(lst_day_ref)

    out_path = get_out_path(os.path.join("Data", "MOD11A1", "3km_nearly_overlapped", area))
    fh_out = Dataset(os.path.join(out_path, doy + ".nc"), "w")

    fh_out.createDimension('lat', n_lat)
    fh_out.createDimension('lon', n_lon)

    for v_name, varin in fh_in.variables.items():
        if v_name == "lat" or v_name == "lon" or v_name == "LST_Day":
            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[:]

    lst_day_mask = ma.getmaskarray(fh_in.variables["LST_Day"][:])
    lst_day_values = fh_in.variables["LST_Day"][:]
    for i in range(n_lat):
        for j in range(n_lon):
            if lst_day_mask[i, j]:
                for k in range(n_days):
                    if not lst_day_ref_mask[k, i, j]:
                        lst_day_values[i, j] = lst_day_ref[k, i, j]
                        break
    fh_out.variables["LST_Day"][:] = lst_day_values[:]







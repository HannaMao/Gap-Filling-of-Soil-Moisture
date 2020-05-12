# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import os
import datetime
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
import shutil

from ..utils import get_out_path


def modis_lai_extract(dates_folder, nc_file):
    out_path = get_out_path(os.path.join("Data", "MCD15A3H", "500m"))

    fh_in = Dataset(os.path.join("n5eil01u.ecs.nsidc.org", "MCD15A3H", dates_folder, nc_file + ".nc"), 'r')

    for index, n_days in enumerate(fh_in.variables['time'][:]):
        date = (datetime.datetime(2000, 1, 1, 0, 0) + datetime.timedelta(int(n_days))).strftime('%Y%m%d')
        print(date)
        fh_out = Dataset(os.path.join(out_path, date + '.nc'), 'w')

        for name, dim in fh_in.dimensions.items():
            if name != 'time':
                fh_out.createDimension(name, len(dim) if not dim.isunlimited() else None)

        ignore_features = ["time", "crs", "FparExtra_QC", "FparLai_QC"]
        mask_value_dic = {'Lai_500m': 10, 'LaiStdDev_500m': 10, 'Fpar_500m': 1, 'FparStdDev_500m': 1}
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
                    vin = varin[index, :, :]
                    vin = ma.masked_greater(vin, mask_value_dic[v_name])
                    vin = ma.masked_less(vin, 0)
                    outVar[:] = vin[:]
        fh_out.close()
    fh_in.close()


def modis_lai_fill_gap(in_path, doy_start, doy_end):
    """
    Fill spatial_temporal_regional_learning gap as MODIS MCD15A3H is a 4-day composite data set.
    Simply copy and paste files
    """
    date_start = datetime.datetime.strptime(doy_start, "%Y%m%d").date()
    date_end = datetime.datetime.strptime(doy_end, "%Y%m%d").date()

    for nc_file in os.listdir(in_path):
        if nc_file.endswith('.nc'):
            nc_date = datetime.datetime.strptime(nc_file[:-3], "%Y%m%d").date()
            if date_start <= nc_date <= date_end:
                print(nc_file, "----",)
                doy = int(datetime.datetime.strptime(nc_file[:-3], '%Y%m%d').strftime('%Y%j'))
                for new_doy in [doy + x for x in range(1, 4)]:
                    shutil.copy2(os.path.join(in_path, nc_file), os.path.join(in_path, '{}.nc'.format(
                        datetime.datetime.strptime(str(new_doy), '%Y%j').strftime('%Y%m%d'))))
                    print('{}.nc'.format(
                        datetime.datetime.strptime(str(new_doy), '%Y%j').strftime('%Y%m%d')),)
                print('\n')





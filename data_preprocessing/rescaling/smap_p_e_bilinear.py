# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..utils import get_out_path, get_lat_lon, match_lat_lon

import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
from datetime import datetime


def smap_p_e_from_9_to_3(doy_start, doy_end):
    in_path = os.path.join("Data", "SMAP_P_E", "usa")
    out_path = get_out_path(os.path.join("Data", "SMAP_P_E", "usa_3km"))

    date_start = datetime.strptime(doy_start, "%Y%m%d").date()
    date_end = datetime.strptime(doy_end, "%Y%m%d").date()

    cont_var_dic = ["soil_moisture", "tb_v_corrected", "freeze_thaw_fraction", "roughness_coefficient",
                    "surface_temperature", "vegetation_opacity", "vegetation_water_content"]

    ratio_dic = {(0, 0): (4, 2, 2, 1), (0, 2): (4, 2, 2, 1), (2, 0): (4, 2, 2, 1), (2, 2): (4, 2, 2, 1),
                 (0, 1): (2, 1), (1, 0): (2, 1), (1, 2): (2, 1), (2, 1): (2, 1),
                 (1, 1): [1]}

    index_dic = {(0, 0): [(0, 0), (-1, 0), (0, -1), (-1, -1)],
                 (0, 2): [(0, 0), (-1, 0), (0, 1), (-1, 1)],
                 (2, 0): [(0, 0), (1, 0), (0, -1), (1, -1)],
                 (2, 2): [(0, 0), (1, 0), (0, 1), (1, 1)],
                 (0, 1): [(0, 0), (-1, 0)],
                 (1, 0): [(0, 0), (0, -1)],
                 (1, 2): [(0, 0), (0, 1)],
                 (2, 1): [(0, 0), (1, 0)],
                 (1, 1): [(0, 0)]}

    lats, lons = get_lat_lon("M03")
    first_flag = True

    for nc_file in os.listdir(in_path):
        if nc_file.endswith('.nc'):
            nc_date = datetime.strptime(nc_file[:-3], "%Y%m%d").date()
            if date_start <= nc_date <= date_end:
                print(nc_file)
                fh_in = Dataset(os.path.join(in_path, nc_file), mode='r')

                if first_flag:
                    i_lat_start, i_lat_end, i_lon_start, i_lon_end = match_lat_lon(lats,
                                                                                   lons,
                                                                                   fh_in.variables["lat"][:],
                                                                                   fh_in.variables["lon"][:],
                                                                                   expand=1)
                    lats = lats[i_lat_start: i_lat_end + 1]
                    lons = lons[i_lon_start: i_lon_end + 1]
                    first_flag = False

                fh_out = Dataset(os.path.join(out_path, nc_file), 'w')

                n_lats, n_lons = fh_in.dimensions["lat"].size, fh_in.dimensions["lon"].size

                dic_var = {}
                for var in cont_var_dic:
                    dic_var[var] = fh_in.variables[var][:]

                fh_out.createDimension('lat', len(lats))
                fh_out.createDimension('lon', len(lons))

                outVar = fh_out.createVariable('lat', 'f4', ('lat',))
                outVar.setncatts({"units": "degree_north"})
                outVar[:] = lats[:]
                outVar = fh_out.createVariable('lon', 'f4', ('lon',))
                outVar.setncatts({"units": "degree_east"})
                outVar[:] = lons[:]

                for v_name, varin in fh_in.variables.items():
                    if v_name in cont_var_dic:
                        outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})

                for idx, lat_var in enumerate(lats):
                    for idy, lon_var in enumerate(lons):
                        for var in cont_var_dic:
                            if dic_var[var][idx // 3, idy // 3] is not ma.masked:
                                interpolated_value = 0
                                sum_ratio = 0
                                for ratio_index, (x_shift, y_shift) in enumerate(index_dic[(idx % 3, idy % 3)]):
                                    if 0 <= idx // 3 + x_shift <= n_lats - 1 \
                                            and 0 <= idy // 3 + y_shift <= n_lons - 1 \
                                            and dic_var[var][idx // 3 + x_shift, idy // 3 + y_shift] is not ma.masked:
                                        ratio = ratio_dic[(idx % 3, idy % 3)][ratio_index]
                                        interpolated_value += dic_var[var][idx // 3 + x_shift, idy // 3 + y_shift] * ratio
                                        sum_ratio += ratio

                                fh_out.variables[var][idx, idy] = interpolated_value / sum_ratio

                fh_in.close()
                fh_out.close()


def resize_to_match_sentinel(doy_start, doy_end):
    print("Resize & change the range of soil moisture to 0.01 - 0.6")
    in_path = os.path.join("Data", "SMAP_P_E", "usa_3km")
    out_path = get_out_path(os.path.join("Data", "SMAP_P_E", "usa_3km_match_sentinel"))

    date_start = datetime.strptime(doy_start, "%Y%m%d").date()
    date_end = datetime.strptime(doy_end, "%Y%m%d").date()

    fh_sentinel = Dataset(os.path.join("Data", "Sentinel", "static_vars_usa_v2.nc"), "r")
    lats = fh_sentinel.variables["lat"][:]
    lons = fh_sentinel.variables["lon"][:]
    fh_sentinel.close()

    first_flag = True
    i_lat_start = i_lat_end = i_lon_start = i_lon_end = 0
    for nc_file in os.listdir(in_path):
        if nc_file.endswith('.nc'):
            nc_date = datetime.strptime(nc_file[:-3], "%Y%m%d").date()
            if date_start <= nc_date <= date_end:
                print(nc_file)
                fh_in = Dataset(os.path.join(in_path, nc_file), mode='r')

                if first_flag:
                    i_lat_start, i_lat_end, i_lon_start, i_lon_end = match_lat_lon(fh_in.variables["lat"][:],
                                                                                   fh_in.variables["lon"][:],
                                                                                   lats,
                                                                                   lons)

                    assert np.array_equal(lats, fh_in.variables["lat"][i_lat_start: i_lat_end + 1])
                    assert np.array_equal(lons, fh_in.variables["lon"][i_lon_start: i_lon_end + 1])
                    first_flag = False

                fh_out = Dataset(os.path.join(out_path, nc_file), 'w')

                fh_out.createDimension('lat', len(lats))
                fh_out.createDimension('lon', len(lons))

                for v_name, varin in fh_in.variables.items():
                    outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    if v_name == "lat":
                        outVar[:] = lats[:]
                    elif v_name == "lon":
                        outVar[:] = lons[:]
                    elif v_name == "soil_moisture":
                        outVar[:] = ma.getdata(varin[i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1])
                    else:
                        outVar[:] = varin[i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1]

                fh_out.variables["soil_moisture"].setncatts({"valid_min": np.array([0.01]).astype('f4'),
                                                             "valid_max": np.array([0.6]).astype('f4')})

                fh_in.close()
                fh_out.close()








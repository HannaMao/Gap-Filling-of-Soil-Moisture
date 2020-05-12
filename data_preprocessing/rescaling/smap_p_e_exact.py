# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause
from data_preprocessing.utils import get_out_path, generate_doy, get_lat_lon

import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma


def smap_p_e_exact_downscale(doy_start, doy_end):
    """
    smap_p_e usa 9km index range: lat: [171:507] lon: [568: 1241], starting from 0, included
    corresponding 3km index range: lat: [513:1523] lon [1704:3725], starting from 0, included
    """
    in_path = os.path.join("Data", "SMAP_P_E", "usa")
    out_path = get_out_path(os.path.join("Data", "SMAP_P_E", "usa_3km_exact"))
    cont_var_dic = ["soil_moisture", "tb_v_corrected", "freeze_thaw_fraction", "roughness_coefficient",
                    "surface_temperature", "vegetation_opacity", "vegetation_water_content", "albedo"]

    for doy in generate_doy(doy_start, doy_end, ""):
        print(doy)
        fh_in = Dataset(os.path.join(in_path, doy + ".nc"), "r")
        fh_out = Dataset(os.path.join(out_path, doy + ".nc"), "w")

        lats, lons = get_lat_lon("M03")
        lats = lats[513: 1524]
        lons = lons[1704: 3726]

        fh_out.createDimension('lat', len(lats))
        fh_out.createDimension('lon', len(lons))
        outVar = fh_out.createVariable('lat', 'f4', ('lat',))
        outVar.setncatts({"units": "degree_north"})
        outVar[:] = lats[:]
        outVar = fh_out.createVariable('lon', 'f4', ('lon',))
        outVar.setncatts({"units": "degree_east"})
        outVar[:] = lons[:]

        datatype = None
        tb_value, ts_value = None, None
        for v_name, varin in fh_in.variables.items():
            if v_name in cont_var_dic:
                outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                varin_value = varin[:]
                varin_value = np.repeat(varin_value, 3, axis=0)
                varin_value = np.repeat(varin_value, 3, axis=1)
                outVar[:] = varin_value[:]
                if v_name == "tb_v_corrected":
                    datatype = varin.datatype
                    tb_value = varin_value[:]
                if v_name == "surface_temperature":
                    ts_value = varin_value[:]

        outVar = fh_out.createVariable("tb_divide_ts", datatype, ("lat", "lon"))
        outVar[:] = ma.divide(tb_value, ts_value)

        fh_in.close()
        fh_out.close()


def exact_downscale_resize_to_match_sentinel(doy_start, doy_end):
    """
    smap_p_e usa 3km index range: lat: [513:1523] lon [1704:3725], starting from 0, included
    smap_ap usa 3km index range: lat: [566:1444] lon: [1767:3662], starting from 0, included
    """
    print("Resize")
    in_path = os.path.join("Data", "SMAP_P_E", "usa_3km_exact")
    out_path = get_out_path(os.path.join("Data", "SMAP_P_E", "usa_3km_exact_match_sentinel"))

    for doy in generate_doy(doy_start, doy_end, ""):
        print(doy)
        fh_in = Dataset(os.path.join(in_path, doy + ".nc"), mode='r')
        fh_out = Dataset(os.path.join(out_path, doy + ".nc"), 'w')

        lats, lons = get_lat_lon("M03")
        lats = lats[566: 1445]
        lons = lons[1767: 3663]

        fh_out.createDimension('lat', len(lats))
        fh_out.createDimension('lon', len(lons))
        outVar = fh_out.createVariable('lat', 'f4', ('lat',))
        outVar.setncatts({"units": "degree_north"})
        outVar[:] = lats[:]
        outVar = fh_out.createVariable('lon', 'f4', ('lon',))
        outVar.setncatts({"units": "degree_east"})
        outVar[:] = lons[:]

        for v_name, varin in fh_in.variables.items():
            if v_name not in ["lat", "lon"]:
                outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                outVar[:] = varin[53:932, 63:1959]

        fh_in.close()
        fh_out.close()


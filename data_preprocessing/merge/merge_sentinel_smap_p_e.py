# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..utils import select_area, get_lat_lon, match_lat_lon, get_out_path

import os
import numpy.ma as ma
import numpy as np
from netCDF4 import Dataset


def merge_sentinel_smap_p_e(doy, lat1, lat2, lon1, lon2, area_name, verbose=True):
    # merge to check the match between sentinel and smap_e
    fh_dic = dict()
    fh_dic["sentinel"] = Dataset(os.path.join("Data", "Sentinel", "usa", doy + ".nc"), "r")
    fh_dic["smap_p_e"] = Dataset(os.path.join("Data", "SMAP_P_E", "usa_3km_match_sentinel", doy + ".nc"), "r")

    out_path = get_out_path(os.path.join("Data", "Merge_Sentinel_SMAP_Passive", area_name))
    fh_out = Dataset(os.path.join(out_path, doy + ".nc"), "w")

    lat_indices, lon_indices = select_area(lat1, lat2, lon1, lon2, "M03")
    lats, lons = get_lat_lon("M03")
    assert (len(lats) != 0 and len(lons) != 0)
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]

    i_lat_start, i_lat_end, i_lon_start, i_lon_end = match_lat_lon(fh_dic["sentinel"].variables["lat"][:],
                                                                   fh_dic["sentinel"].variables["lon"][:],
                                                                   lats,
                                                                   lons)

    fh_out.createDimension('lat', len(lats))
    fh_out.createDimension('lon', len(lons))

    outVar = fh_out.createVariable('lat', 'f4', ('lat'))
    outVar.setncatts({"units": "degree_north"})
    outVar[:] = lats[:]
    outVar = fh_out.createVariable('lon', 'f4', ('lon'))
    outVar.setncatts({"units": "degree_east"})
    outVar[:] = lons[:]

    rename_dic = {"soil_moisture": "smap_p_e_soil_moisture",
                  "tb_v_corrected": "smap_p_e_tb_v_corrected"}
    ma_dic = {}
    for fName in fh_dic:
        for v_name, varin in fh_dic[fName].variables.items():
            if v_name in ["soil_moisture", "tb_v_corrected", "tb_v_disaggregated"]:
                if fName == "smap_p_e":
                    v_name = rename_dic[v_name]
                outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                outVar[:] = varin[i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1]
                ma_dic[v_name] = ma.getmaskarray(varin[i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1])

    daily_mask = np.logical_or.reduce(list(ma_dic.values()))
    if verbose:
        print("Before mask, number of valid grids:", fh_out.variables["soil_moisture"][:].count())
        for var in fh_out.variables:
            if var != "lat" and var != "lon":
                if ma.is_masked(fh_out.variables[var][:]):
                    print(var, ma.array(fh_out.variables["soil_moisture"][:], mask=
                    ma.mask_or(ma.getmaskarray(fh_out.variables["soil_moisture"][:]),
                               ma.getmaskarray(fh_out.variables[var][:]))).count())
    for var in fh_out.variables:
        if var != "lat" and var != "lon":
            fh_out.variables[var][:] = ma.array(fh_out.variables[var][:], mask=daily_mask)
    if verbose:
        print("After mask, number of valid grids:", fh_out.variables["soil_moisture"][:].count())
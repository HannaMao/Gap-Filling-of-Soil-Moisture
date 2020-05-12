# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..utils import select_area, get_lat_lon, match_lat_lon, get_out_path, obtain_unique_landcover_by_state
from .. import landcover_class_dic, states_index_dic
from ..analysis import check_dominated_lc

import os
import numpy.ma as ma
import numpy as np
from netCDF4 import Dataset


def merge_various_variables(sentinel_path, out_path, out_file, lat1, lat2, lon1, lon2, area_name, doy, n_hist,
                            ignore_fields=list()):
    print(area_name, doy, out_file)
    fh_dic = dict()
    fh_dic["sentinel"] = Dataset(os.path.join(sentinel_path, doy + ".nc"), "r")
    fh_dic["sentinel_9km"] = Dataset(os.path.join("Data", "Sentinel", "usa_db_9km_only", doy + ".nc"), "r")
    fh_dic["sentinel_landcover"] = Dataset(os.path.join("Data", "Sentinel", "landcover_class_usa.nc"), "r")
    fh_dic["sentinel_surface_flags"] = Dataset(os.path.join("Data", "Sentinel", "static_surface_flags_usa.nc"), "r")
    fh_dic["sentinel_static_vars"] = Dataset(os.path.join("Data", "Sentinel", "static_vars_usa_v2.nc"), "r")
    rb_hist = "sentinel_rb_hist_" + str(n_hist)
    fh_dic[rb_hist] = Dataset(os.path.join("Data", "Sentinel", "usa_rb_hist_average_" + str(n_hist), doy + ".nc"), "r")
    fh_dic["smap_p_e"] = Dataset(os.path.join("Data", "SMAP_P_E", "usa_3km_exact_match_sentinel", doy + ".nc"), "r")
    fh_dic["lai"] = Dataset(os.path.join("Data", "MCD15A3H", "3km", doy + ".nc"), "r")
    fh_dic["lst"] = Dataset(os.path.join("Data", "MOD11A1", "3km_nearly_overlapped", area_name, doy + ".nc"), "r")
    fh_dic["gpm"] = Dataset(os.path.join("Data", "GPM", "hist_added", doy + ".nc"), "r")

    fh_dic["soil"] = Dataset(os.path.join("Data", "SoilType", "3km", "soiltype_usa.nc"), "r")
    fh_dic["ele"] = Dataset(os.path.join("Data", "Elevation", "3km", "elevation_usa.nc"), "r")
    fh_dic["bulk_density"] = Dataset(os.path.join("Data", "Bulk_Density", "bulk_density_3km.nc"), "r")

    out_path = get_out_path(out_path)
    fh_out = Dataset(os.path.join(out_path, out_file), "w")

    lat_indices, lon_indices = select_area(lat1, lat2, lon1, lon2, "M03")
    lats, lons = get_lat_lon("M03")
    assert (len(lats) != 0 and len(lons) != 0)
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]

    i_lat_start, i_lat_end, i_lon_start, i_lon_end = match_lat_lon(fh_dic["sentinel_landcover"].variables["lat"][:],
                                                                   fh_dic["sentinel_landcover"].variables["lon"][:],
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
        if fName == "lst":
            varin = fh_dic[fName].variables["LST_Day"]
            outVar = fh_out.createVariable("LST_Day", varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            varin_value = ma.masked_invalid(ma.asarray(varin[i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1]))
            outVar[:] = varin_value
            ma_dic["lst"] = ma.getmaskarray(varin_value)
        else:
            for v_name, varin in fh_dic[fName].variables.items():
                if v_name != "lat" and v_name != "lon" and v_name not in ignore_fields:
                    if fName == "smap_p_e":
                        v_name = rename_dic[v_name]
                    outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    varin_value = ma.masked_invalid(
                        ma.asarray(varin[i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1]))
                    outVar[:] = varin_value
                    ma_dic[v_name] = ma.getmaskarray(varin_value)

    daily_mask = np.logical_or.reduce(list(ma_dic.values()))
    print("Before mask, number of valid grids:", fh_out.variables["soil_moisture"][:].count())
    # for var in fh_out.variables:
    #     if var != "lat" and var != "lon":
    #         if ma.is_masked(fh_out.variables[var][:]):
    #             print(var, ma.array(fh_out.variables["soil_moisture"][:], mask=
    #             ma.mask_or(ma.getmaskarray(fh_out.variables["soil_moisture"][:]),
    #                        ma.getmaskarray(fh_out.variables[var][:]))).count())

    for var in fh_out.variables:
        if var != "lat" and var != "lon":
            fh_out.variables[var][:] = ma.array(fh_out.variables[var][:], mask=daily_mask)

    # construct binary features from landcover class
    for landcover in obtain_unique_landcover_by_state(states_index_dic[area_name]):
        varin = fh_dic["sentinel_landcover"].variables["landcover_class"]
        outVar = fh_out.createVariable("_".join(["landcover", landcover_class_dic[landcover]]),
                                       varin.datatype, varin.dimensions)
        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
        outVar[:] = ma.array(np.full((len(lats), len(lons)), 0), mask=daily_mask)

    landcover_values = fh_dic["sentinel_landcover"].variables["landcover_class"][i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1]
    for i in range(len(lats)):
        for j in range(len(lons)):
            if not daily_mask[i, j]:
                set_variable = "landcover_" + landcover_class_dic[landcover_values[i, j]]
                fh_out.variables[set_variable][i, j] = 1

    print("After mask, number of valid grids:", fh_out.variables["soil_moisture"][:].count())


def merge_various_variables_v2(sentinel_path, out_path, out_file, lat1, lat2, lon1, lon2, area_name, doy, n_hist,
                               selected_sentinel_fields):
    print(area_name, doy, out_file)
    fh_dic = dict()
    fh_dic["sentinel"] = Dataset(os.path.join(sentinel_path, doy + ".nc"), "r")
    rb_hist = "sentinel_rb_hist_" + str(n_hist)
    fh_dic[rb_hist] = Dataset(os.path.join("Data", "Sentinel", "usa_rb_hist_average_" + str(n_hist), doy + ".nc"), "r")
    fh_dic["smap_p_e"] = Dataset(os.path.join("Data", "SMAP_P_E", "usa_3km_exact_match_sentinel", doy + ".nc"), "r")

    fh_dic["lai"] = Dataset(os.path.join("Data", "MCD15A3H", "3km", doy + ".nc"), "r")
    fh_dic["lst"] = Dataset(os.path.join("Data", "MOD11A1", "3km_nearly_overlapped", area_name, doy + ".nc"), "r")
    fh_dic["gpm"] = Dataset(os.path.join("Data", "GPM", "hist_added", doy + ".nc"), "r")

    fh_dic["landcover"] = Dataset(os.path.join("Data", "LANDCOVER", "landcover_3km_usa_2016.nc"), "r")
    fh_dic["sentinel_surface_flags"] = Dataset(os.path.join("Data", "Sentinel", "static_surface_flags_usa.nc"), "r")
    fh_dic["sentinel_static_vars"] = Dataset(os.path.join("Data", "Sentinel", "static_vars_usa_v2.nc"), "r")
    fh_dic["soil"] = Dataset(os.path.join("Data", "Soil_Fraction", "soil_fraction_3km_usa.nc"), "r")
    fh_dic["ele"] = Dataset(os.path.join("Data", "Elevation", "3km", "elevation_usa.nc"), "r")
    fh_dic["bulk_density"] = Dataset(os.path.join("Data", "Bulk_Density", "bulk_density_3km_usa.nc"), "r")
    fh_dic["slope"] = Dataset(os.path.join("Data", "Elevation", "slope.nc"), "r")

    out_path = get_out_path(out_path)
    fh_out = Dataset(os.path.join(out_path, out_file), "w")

    lat_indices, lon_indices = select_area(lat1, lat2, lon1, lon2, "M03")
    lats, lons = get_lat_lon("M03")
    assert (len(lats) != 0 and len(lons) != 0)
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]

    i_lat_start, i_lat_end, i_lon_start, i_lon_end = match_lat_lon(fh_dic["landcover"].variables["lat"][:],
                                                                   fh_dic["landcover"].variables["lon"][:],
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

    ma_dic = {}
    lc_ids, lc_names = check_dominated_lc(lat1=lat1, lat2=lat2, lon1=lon1, lon2=lon2, reso="3km")
    for fName in fh_dic:
        for v_name, varin in fh_dic[fName].variables.items():
            if v_name != "lat" and v_name != "lon":
                v_name = "smap_p_e_" + v_name if fName == "smap_p_e" else v_name
                if fName == "sentinel" and v_name not in selected_sentinel_fields:
                    continue
                if fName == "landcover" and v_name[3:] not in lc_names:
                    continue
                outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                varin_value = ma.masked_invalid(
                    ma.asarray(varin[i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1]))
                outVar[:] = varin_value
                ma_dic[v_name] = ma.getmaskarray(varin_value)

    daily_mask = np.logical_or.reduce(list(ma_dic.values()))
    print("Before mask, number of valid grids:", fh_out.variables["soil_moisture"][:].count())
    # for var in fh_out.variables:
    #     if var != "lat" and var != "lon":
    #         if ma.is_masked(fh_out.variables[var][:]):
    #             print(var, ma.array(fh_out.variables["soil_moisture"][:], mask=
    #             ma.mask_or(ma.getmaskarray(fh_out.variables["soil_moisture"][:]),
    #                        ma.getmaskarray(fh_out.variables[var][:]))).count())

    for var in fh_out.variables:
        if var != "lat" and var != "lon":
            fh_out.variables[var][:] = ma.array(fh_out.variables[var][:], mask=daily_mask)

    print("After mask, number of valid grids:", fh_out.variables["soil_moisture"][:].count())


def merge_various_variables_usa(sentinel_path, out_path, out_file, doy, n_hist, ignore_fields=list()):
    print(doy, out_file)
    fh_dic = dict()
    fh_dic["sentinel"] = Dataset(os.path.join(sentinel_path, doy + ".nc"), "r")
    fh_dic["sentinel_9km"] = Dataset(os.path.join("Data", "Sentinel", "usa_db_9km_only", doy + ".nc"), "r")
    fh_dic["sentinel_landcover"] = Dataset(os.path.join("Data", "Sentinel", "landcover_class_usa.nc"), "r")
    fh_dic["sentinel_surface_flags"] = Dataset(os.path.join("Data", "Sentinel", "static_surface_flags_usa.nc"), "r")
    fh_dic["sentinel_static_vars"] = Dataset(os.path.join("Data", "Sentinel", "static_vars_usa_v2.nc"), "r")
    rb_hist = "sentinel_rb_hist_" + str(n_hist)
    fh_dic[rb_hist] = Dataset(os.path.join("Data", "Sentinel", "usa_rb_hist_average_" + str(n_hist), doy + ".nc"), "r")
    fh_dic["smap_p_e"] = Dataset(os.path.join("Data", "SMAP_P_E", "usa_3km_exact_match_sentinel", doy + ".nc"), "r")
    fh_dic["lai"] = Dataset(os.path.join("Data", "MCD15A3H", "3km", doy + ".nc"), "r")
    fh_dic["lst"] = Dataset(os.path.join("Data", "MOD11A1", "3km_nearly_overlapped", "usa", doy + ".nc"), "r")
    fh_dic["gpm"] = Dataset(os.path.join("Data", "GPM", "hist_added", doy + ".nc"), "r")

    fh_dic["soil"] = Dataset(os.path.join("Data", "SoilType", "3km", "soiltype_usa.nc"), "r")
    fh_dic["ele"] = Dataset(os.path.join("Data", "Elevation", "3km", "elevation_usa.nc"), "r")
    fh_dic["bulk_density"] = Dataset(os.path.join("Data", "Bulk_Density", "bulk_density_3km.nc"), "r")

    out_path = get_out_path(out_path)
    fh_out = Dataset(os.path.join(out_path, out_file), "w")

    dim_dic = {}
    for name, dim in fh_dic["sentinel"].dimensions.items():
        fh_out.createDimension(name, len(dim))
    for v_name, varin in fh_dic["sentinel"].variables.items():
        if v_name == "lat" or v_name == "lon":
            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[:]
            dim_dic[v_name] = varin[:]

    rename_dic = {"soil_moisture": "smap_p_e_soil_moisture",
                  "tb_v_corrected": "smap_p_e_tb_v_corrected"}
    ma_dic = {}
    for fName in fh_dic:
        if fName == "lst":
            varin = fh_dic[fName].variables["LST_Day"]
            outVar = fh_out.createVariable("LST_Day", varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            varin_value = ma.masked_invalid(ma.asarray(varin[:]))
            outVar[:] = varin_value
            ma_dic["lst"] = ma.getmaskarray(varin_value)
        else:
            for v_name, varin in fh_dic[fName].variables.items():
                if v_name != "lat" and v_name != "lon" and v_name not in ignore_fields:
                    if fName == "smap_p_e":
                        v_name = rename_dic[v_name]
                    outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    varin_value = ma.masked_invalid(ma.asarray(varin[:]))
                    outVar[:] = varin_value
                    ma_dic[v_name] = ma.getmaskarray(varin_value)

    daily_mask = np.logical_or.reduce(list(ma_dic.values()))
    print("Before mask, number of valid grids:", fh_out.variables["soil_moisture"][:].count())
    # for var in fh_out.variables:
    #     if var != "lat" and var != "lon":
    #         if ma.is_masked(fh_out.variables[var][:]):
    #             print(var, ma.array(fh_out.variables["soil_moisture"][:], mask=
    #             ma.mask_or(ma.getmaskarray(fh_out.variables["soil_moisture"][:]),
    #                        ma.getmaskarray(fh_out.variables[var][:]))).count())

    for var in fh_out.variables:
        if var != "lat" and var != "lon":
            fh_out.variables[var][:] = ma.array(fh_out.variables[var][:], mask=daily_mask)

    # construct binary features from landcover class
    for landcover in land_cover_class_dic.values():
        varin = fh_dic["sentinel_landcover"].variables["landcover_class"]
        outVar = fh_out.createVariable("_".join(["landcover", landcover]),
                                       varin.datatype, varin.dimensions)
        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
        outVar[:] = ma.array(np.full((len(dim_dic["lat"]), len(dim_dic["lon"])), 0), mask=daily_mask)

    landcover_values = fh_dic["sentinel_landcover"].variables["landcover_class"][:]
    for i in range(len(dim_dic["lat"])):
        for j in range(len(dim_dic["lon"])):
            if not daily_mask[i, j]:
                set_variable = "landcover_" + land_cover_class_dic[landcover_values[i, j]]
                fh_out.variables[set_variable][i, j] = 1

    print("After mask, number of valid grids:", fh_out.variables["soil_moisture"][:].count())












# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..utils import get_out_path, get_lat_lon, select_area, generate_nearest_doys, match_lat_lon, generate_doy

import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
from operator import and_
from collections import defaultdict

ignore_vars = ["EASE_column_index_3km", "EASE_row_index_3km", "spacecraft_overpass_time_seconds_3km",
               "disaggregated_tb_v_qual_flag_3km", "retrieval_qual_flag_3km",
               "SMAP_Sentinel_overpass_timediff_hr_3km", "landcover_class_3km", "surface_flag_3km",
               "soil_moisture_std_dev_3km", "tb_v_disaggregated_std_3km"]


def generate_input_file_list(doy):
    """
    Keep the most recent version if a file has been process multiple times.
    """
    patches_dic = defaultdict(list)
    input_files = []

    check_path = os.path.join("n5eil01u.ecs.nsidc.org", "SMAP", "SPL2SMAP_S.002", doy)
    if os.path.isdir(check_path):
        for f in os.listdir(check_path):
            if f.endswith(".h5"):
                patches_dic[f[:-7]].append(f)

        for _, value in patches_dic.items():
            input_files.append(sorted(value)[-1])

        return input_files
    else:
        return input_files


def extract_static_vars_local(lat1, lat2, lon1, lon2, area_name, var_list, doy_start, doy_end):
    lat_indices, lon_indices = select_area(lat1, lat2, lon1, lon2, "M03")
    lats, lons = get_lat_lon("M03")
    assert (len(lats) != 0 and len(lons) != 0)
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]

    out_path = get_out_path(os.path.join("Data", "Sentinel"))
    fh_out = Dataset(os.path.join(out_path, "static_vars_" + area_name + ".nc"), "w")
    fh_out.createDimension("lat", len(lats))
    fh_out.createDimension("lon", len(lons))

    general_in_path = os.path.join("n5eil01u.ecs.nsidc.org", "SMAP", "SPL2SMAP_S.002")
    first_flag = True

    for doy_folder in generate_doy(doy_start, doy_end, "."):
        if doy_folder in os.listdir(general_in_path):
            for f in os.listdir(os.path.join(general_in_path, doy_folder)):
                if f.endswith(".h5"):
                    fh_in = Dataset(os.path.join(general_in_path, doy_folder, f), "r")
                    group_3km = fh_in.groups["Soil_Moisture_Retrieval_Data_3km"]

                    lat_start = group_3km.variables["EASE_row_index_3km"][0, 0]
                    lat_end = group_3km.variables["EASE_row_index_3km"][-1, 0]
                    lon_start = group_3km.variables["EASE_column_index_3km"][0, 0]
                    lon_end = group_3km.variables["EASE_column_index_3km"][0, -1]

                    if lat_end <= lat_indices[0] or lat_start >= lat_indices[1] \
                            or lon_end <= lon_indices[0] or lon_start >= lon_indices[1]:
                        fh_in.close()
                        continue

                    print(f)
                    if first_flag:
                        for v_name, varin in group_3km.variables.items():
                            if v_name in ["latitude_3km", "longitude_3km"]:
                                outVar = fh_out.createVariable(v_name[:3], varin.datatype, (v_name[:3]))
                            elif v_name in var_list:
                                outVar = fh_out.createVariable(v_name[:-4], varin.datatype, ("lat", "lon"))
                            else:
                                continue
                            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                        fh_out.variables["lat"][:] = lats[:]
                        fh_out.variables["lon"][:] = lons[:]

                        first_flag = False

                    out_lat_start = max(lat_start - lat_indices[0], 0)
                    out_lat_end = min(lat_end + 1 - lat_indices[0], len(lats))
                    out_lon_start = max(lon_start - lon_indices[0], 0)
                    out_lon_end = min(lon_end + 1 - lon_indices[0], len(lons))
                    in_lat_start = max(lat_indices[0] - lat_start, 0)
                    in_lat_end = min(lat_indices[1] - lat_start, lat_end - lat_start + 1)
                    in_lon_start = max(lon_indices[0] - lon_start, 0)
                    in_lon_end = min(lon_indices[1] - lon_start, lon_end - lon_start + 1)
                    assert (out_lat_end - out_lat_start == in_lat_end - in_lat_start)
                    assert (out_lon_end - out_lon_start == in_lon_end - in_lon_start)
                    for v_name, varin in group_3km.variables.items():
                        if v_name in var_list:
                            a = fh_out.variables[v_name[:-4]][out_lat_start: out_lat_end, out_lon_start: out_lon_end]
                            b = varin[in_lat_start: in_lat_end, in_lon_start: in_lon_end]
                            if not isinstance(a, ma.MaskedArray):
                                a = ma.array(a, mask=np.zeros(a.shape))
                            if not isinstance(b, ma.MaskedArray):
                                b = ma.array(b, mask=np.zeros(b.shape))

                            for i in range(0, out_lat_end - out_lat_start):
                                for j in range(0, out_lon_end - out_lon_start):
                                    if not ma.getmaskarray(a)[i, j] and not ma.getmaskarray(b)[i, j]:
                                        assert a[i, j] == b[i, j], v_name + " " + str(a[i, j]) + " " + str(b[i, j])
                                    elif ma.getmaskarray(a)[i, j] and not ma.getmaskarray(b)[i, j]:
                                        a[i, j] = b[i, j]
                            fh_out.variables[v_name[:-4]][out_lat_start: out_lat_end, out_lon_start: out_lon_end] = a

                    fh_in.close()
    fh_out.close()


def combine_to_3km(doy):
    in_path = os.path.join("n5eil01u.ecs.nsidc.org", "SMAP", "SPL2SMAP_S.002", doy)
    out_path = get_out_path(os.path.join("Data", "Sentinel", "combined_3km"))

    print(doy)
    fh_out = Dataset(os.path.join(out_path, doy.replace(".", "") + ".nc"), "w")
    fh_out.createDimension("lat", 4872)
    fh_out.createDimension("lon", 11568)
    lats, lons = get_lat_lon("M03")
    assert (len(lats) != 0 and len(lons) != 0)

    count_dic = {}
    first_flag = True
    for f in generate_input_file_list(doy):
        if f.endswith(".h5"):
            fh_in = Dataset(os.path.join(in_path, f), "r")
            group_3km = fh_in.groups["Soil_Moisture_Retrieval_Data_3km"]

            lat_start = group_3km.variables["EASE_row_index_3km"][0, 0]
            lat_end = group_3km.variables["EASE_row_index_3km"][-1, 0]
            lon_start = group_3km.variables["EASE_column_index_3km"][0, 0]
            lon_end = group_3km.variables["EASE_column_index_3km"][0, -1]

            if first_flag:
                for v_name, varin in group_3km.variables.items():
                    if v_name in ["latitude_3km", "longitude_3km"]:
                        outVar = fh_out.createVariable(v_name[:3], varin.datatype, (v_name[:3]))
                    elif not v_name.endswith("apm_3km") and v_name not in ignore_vars:
                        outVar = fh_out.createVariable(v_name[:-4], varin.datatype, ("lat", "lon"))
                        count_dic[v_name[:-4]] = np.zeros((len(lats), len(lons)))
                    else:
                        continue
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                fh_out.variables["lat"][:] = lats[:]
                fh_out.variables["lon"][:] = lons[:]

                first_flag = False

            for v_name, varin in group_3km.variables.items():
                if v_name not in ["latitude_3km", "longitude_3km"] \
                        and not v_name.endswith("apm_3km") \
                        and v_name not in ignore_vars:
                    assert varin[:].shape == ((lat_end - lat_start + 1), (lon_end - lon_start + 1))
                    a = fh_out.variables[v_name[:-4]][lat_start: lat_end + 1, lon_start: lon_end + 1]
                    b = varin[:]
                    if not isinstance(a, ma.MaskedArray):
                        a = ma.array(a, mask=np.zeros(a.shape))
                    if not isinstance(b, ma.MaskedArray):
                        b = ma.array(b, mask=np.zeros(b.shape))
                    fh_out.variables[v_name[:-4]][lat_start: lat_end + 1, lon_start: lon_end + 1] = \
                        ma.array(ma.filled(a, 0) + ma.filled(b, 0), mask=ma.array([*map(and_, ma.getmaskarray(a),
                                                                                        ma.getmaskarray(b))]))
                    count_dic[v_name[:-4]][lat_start: lat_end + 1, lon_start: lon_end + 1] += ~ma.getmask(b)

            fh_in.close()

    for var in count_dic:
        fh_out.variables[var][:] = np.divide(fh_out.variables[var][:], count_dic[var])

    fh_out.close()


def filter_files(input_files, area_filter):
    area_file = {}
    for area in area_filter:
        area_file[area] = {}
        for t in area_filter[area]:
            area_file[area][t] = set()

    res = []
    for f in input_files:
        for area in area_filter:
            for t in area_filter[area]:
                if (f.split("_")[4], f.split("_")[-3]) in area_filter[area][t]:
                    res.append(f)
                    area_file[area][t].add(f.split("_")[5][:8])

    return res, area_file


def combine_to_3km_local(doy, lat1, lat2, lon1, lon2, area_name, convert_to_db, with_filter=False, area_filter={}):
    """
    left-up: lat1 lon1  left-down: lat2 lon1  right-up: lat1 lon2  right-down: lat2 lon2
    e.g.
    For United States http://en.wikipedia.org/wiki/Extreme_points_of_the_United_States#Westernmost
    :param lat1: 50
    :param lat2: 24
    :param lon1: -125
    :param lon2: -66
    """
    lat_indices, lon_indices = select_area(lat1, lat2, lon1, lon2, "M03")
    lats, lons = get_lat_lon("M03")
    assert (len(lats) != 0 and len(lons) != 0)
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]

    in_path = os.path.join("n5eil01u.ecs.nsidc.org", "SMAP", "SPL2SMAP_S.002", doy)
    out_path = get_out_path(os.path.join("Data", "Sentinel", area_name))
    print(doy)

    count_dic = {}
    value_dic = {}
    area_file = {}
    first_flag = True
    input_files = generate_input_file_list(doy)
    if with_filter:
        input_files, area_file = filter_files(input_files, area_filter)
    if len(input_files) == 0:
        print("No files for {}".format(doy))
        return {}
    else:
        fh_out = Dataset(os.path.join(out_path, doy.replace(".", "") + ".nc"), "w")
        fh_out.createDimension("lat", len(lats))
        fh_out.createDimension("lon", len(lons))
        for f in input_files:
            if f.endswith(".h5"):
                fh_in = Dataset(os.path.join(in_path, f), "r")
                group_3km = fh_in.groups["Soil_Moisture_Retrieval_Data_3km"]

                lat_start = group_3km.variables["EASE_row_index_3km"][0, 0]
                lat_end = group_3km.variables["EASE_row_index_3km"][-1, 0]
                lon_start = group_3km.variables["EASE_column_index_3km"][0, 0]
                lon_end = group_3km.variables["EASE_column_index_3km"][0, -1]

                if lat_end <= lat_indices[0] or lat_start >= lat_indices[1] \
                        or lon_end <= lon_indices[0] or lon_start >= lon_indices[1]:
                    fh_in.close()
                    continue

                if first_flag:
                    for v_name, varin in group_3km.variables.items():
                        if v_name in ["latitude_3km", "longitude_3km"]:
                            outVar = fh_out.createVariable(v_name[:3], varin.datatype, (v_name[:3]))
                        elif not v_name.endswith("apm_3km") and v_name not in ignore_vars:
                            outVar = fh_out.createVariable(v_name[:-4], varin.datatype, ("lat", "lon"))
                            count_dic[v_name[:-4]] = np.zeros((len(lats), len(lons)))
                            value_dic[v_name[:-4]] = np.zeros((len(lats), len(lons)))
                        else:
                            continue
                        if v_name[:-4] not in ["sigma0_vv_aggregated", "sigma0_vh_aggregated"]:
                            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    fh_out.variables["lat"][:] = lats[:]
                    fh_out.variables["lon"][:] = lons[:]

                    first_flag = False

                out_lat_start = max(lat_start - lat_indices[0], 0)
                out_lat_end = min(lat_end + 1 - lat_indices[0], len(lats))
                out_lon_start = max(lon_start - lon_indices[0], 0)
                out_lon_end = min(lon_end + 1 - lon_indices[0], len(lons))
                in_lat_start = max(lat_indices[0] - lat_start, 0)
                in_lat_end = min(lat_indices[1] - lat_start, lat_end - lat_start + 1)
                in_lon_start = max(lon_indices[0] - lon_start, 0)
                in_lon_end = min(lon_indices[1] - lon_start, lon_end - lon_start + 1)
                assert (out_lat_end - out_lat_start == in_lat_end - in_lat_start)
                assert (out_lon_end - out_lon_start == in_lon_end - in_lon_start)
                for v_name, varin in group_3km.variables.items():
                    if v_name not in ["latitude_3km", "longitude_3km"] \
                            and not v_name.endswith("apm_3km") \
                            and v_name not in ignore_vars:
                        a = value_dic[v_name[:-4]][out_lat_start: out_lat_end, out_lon_start: out_lon_end]
                        b = varin[in_lat_start: in_lat_end, in_lon_start: in_lon_end]
                        if not isinstance(a, ma.MaskedArray):
                            a = ma.array(a, mask=np.zeros(a.shape))
                        if not isinstance(b, ma.MaskedArray):
                            b = ma.array(b, mask=np.zeros(b.shape))
                        value_dic[v_name[:-4]][out_lat_start: out_lat_end, out_lon_start: out_lon_end] = \
                            ma.array(ma.filled(a, 0) + ma.filled(b, 0), mask=ma.array([*map(and_, ma.getmaskarray(a),
                                                                                            ma.getmaskarray(b))]))
                        count_dic[v_name[:-4]][out_lat_start: out_lat_end, out_lon_start: out_lon_end] += ~ma.getmask(b)

                fh_in.close()

        for var in value_dic:
            averaged_values = np.divide(ma.masked_equal(value_dic[var][:], 0), count_dic[var])
            if var not in ["sigma0_vv_aggregated", "sigma0_vh_aggregated"]:
                fh_out.variables[var][:] = averaged_values[:]
            else:
                if convert_to_db:
                    fh_out.variables[var][:] = 10 * ma.log(averaged_values[:]) / ma.log(10)
                else:
                    fh_out.variables[var][:] = averaged_values[:]
        fh_out.variables["soil_moisture"].setncatts({"valid_min": np.array([0.01]).astype('f4'),
                                                     "valid_max": np.array([0.6]).astype('f4')})

        fh_out.close()
        return area_file


def reorganize_to_sentinel_only_local(doy, lat1, lat2, lon1, lon2, area_name):
    """
        left-up: lat1 lon1  left-down: lat2 lon1  right-up: lat1 lon2  right-down: lat2 lon2
        e.g.
        For United States http://en.wikipedia.org/wiki/Extreme_points_of_the_United_States#Westernmost
        :param lat1: 50
        :param lat2: 24
        :param lon1: -125
        :param lon2: -66
        """
    var_list = ["sigma0_vh_aggregated_3km", "sigma0_vv_aggregated_3km"]

    lat_indices, lon_indices = select_area(lat1, lat2, lon1, lon2, "M03")
    lats, lons = get_lat_lon("M03")
    assert (len(lats) != 0 and len(lons) != 0)
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]

    out_path = get_out_path(os.path.join("Data", "Sentinel", "sentinel_only", area_name))
    fh_out = Dataset(os.path.join(out_path, doy + ".nc"), "w")
    fh_out.createDimension("lat", len(lats))
    fh_out.createDimension("lon", len(lons))

    count_dic = {}
    first_flag = True
    check_doys = generate_nearest_doys(doy, 5, ".")
    for c_doy in check_doys:
        in_path = os.path.join("n5eil01u.ecs.nsidc.org", "SMAP", "SPL2SMAP_S.002", c_doy)
        if os.path.isdir(in_path):
            for f in os.listdir(in_path):
                if f[24:32] == doy and f.endswith(".h5"):
                    fh_in = Dataset(os.path.join(in_path, f), "r")
                    group_3km = fh_in.groups["Soil_Moisture_Retrieval_Data_3km"]

                    lat_start = group_3km.variables["EASE_row_index_3km"][0, 0]
                    lat_end = group_3km.variables["EASE_row_index_3km"][-1, 0]
                    lon_start = group_3km.variables["EASE_column_index_3km"][0, 0]
                    lon_end = group_3km.variables["EASE_column_index_3km"][0, -1]

                    if lat_end <= lat_indices[0] or lat_start >= lat_indices[1] \
                            or lon_end <= lon_indices[0] or lon_start >= lon_indices[1]:
                        fh_in.close()
                        continue

                    print(f)
                    if first_flag:
                        for v_name, varin in group_3km.variables.items():
                            if v_name in ["latitude_3km", "longitude_3km"]:
                                outVar = fh_out.createVariable(v_name[:3], varin.datatype, (v_name[:3]))
                            elif v_name in var_list:
                                outVar = fh_out.createVariable(v_name[:-4], varin.datatype, ("lat", "lon"))
                                count_dic[v_name[:-4]] = np.zeros((len(lats), len(lons)))
                            else:
                                continue
                            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                        fh_out.variables["lat"][:] = lats[:]
                        fh_out.variables["lon"][:] = lons[:]

                        first_flag = False

                    out_lat_start = max(lat_start - lat_indices[0], 0)
                    out_lat_end = min(lat_end + 1 - lat_indices[0], len(lats))
                    out_lon_start = max(lon_start - lon_indices[0], 0)
                    out_lon_end = min(lon_end + 1 - lon_indices[0], len(lons))
                    in_lat_start = max(lat_indices[0] - lat_start, 0)
                    in_lat_end = min(lat_indices[1] - lat_start, lat_end - lat_start + 1)
                    in_lon_start = max(lon_indices[0] - lon_start, 0)
                    in_lon_end = min(lon_indices[1] - lon_start, lon_end - lon_start + 1)
                    assert (out_lat_end - out_lat_start == in_lat_end - in_lat_start)
                    assert (out_lon_end - out_lon_start == in_lon_end - in_lon_start)
                    for v_name, varin in group_3km.variables.items():
                        if v_name in var_list:
                            a = fh_out.variables[v_name[:-4]][out_lat_start: out_lat_end, out_lon_start: out_lon_end]
                            b = varin[in_lat_start: in_lat_end, in_lon_start: in_lon_end]
                            if not isinstance(a, ma.MaskedArray):
                                a = ma.array(a, mask=np.zeros(a.shape))
                            if not isinstance(b, ma.MaskedArray):
                                b = ma.array(b, mask=np.zeros(b.shape))
                            fh_out.variables[v_name[:-4]][out_lat_start: out_lat_end, out_lon_start: out_lon_end] = \
                                ma.array(ma.filled(a, 0) + ma.filled(b, 0), mask=ma.array([*map(and_, a.mask, b.mask)]))
                            count_dic[v_name[:-4]][out_lat_start: out_lat_end,
                            out_lon_start: out_lon_end] += ~ma.getmask(b)

                    fh_in.close()

    for var in count_dic:
        fh_out.variables[var][:] = np.divide(fh_out.variables[var][:], count_dic[var])

    fh_out.close()


def _one_to_one_extract_surface_flag_only_local(doy, lat1, lat2, lon1, lon2, area_name):
    """
        left-up: lat1 lon1  left-down: lat2 lon1  right-up: lat1 lon2  right-down: lat2 lon2
        e.g.
        For United States http://en.wikipedia.org/wiki/Extreme_points_of_the_United_States#Westernmost
        :param lat1: 50
        :param lat2: 24
        :param lon1: -125
        :param lon2: -66
        """
    flag_dic = {0: "static_water_body_flag", 2: "coastal_mask_flag", 3: "urban_area_flag",
                4: "precipitation_flag", 5: "snow_or_ice_flag", 6: "permanent_snow_or_ice_flag",
                7: "frozen_ground_flag", 8: "frozen_ground_st_based",
                9: "mountainous_terrain_flag", 10: "dense_vegetation_flag",
                11: "edge_cell_flag", 12: "anomalous_sigma0_flag"}

    lat_indices, lon_indices = select_area(lat1, lat2, lon1, lon2, "M03")
    global_lats, global_lons = get_lat_lon("M03")
    assert (len(global_lats) != 0 and len(global_lons) != 0)

    in_path = os.path.join("n5eil01u.ecs.nsidc.org", "SMAP", "SPL2SMAP_S.002", doy)
    out_path = get_out_path(os.path.join("Data", "Sentinel", "surface_flags", area_name, doy))

    for f in os.listdir(in_path):
        if f.endswith(".h5"):
            fh_in = Dataset(os.path.join(in_path, f), "r")
            group_3km = fh_in.groups["Soil_Moisture_Retrieval_Data_3km"]

            lat_start = group_3km.variables["EASE_row_index_3km"][0, 0]
            lat_end = group_3km.variables["EASE_row_index_3km"][-1, 0]
            lon_start = group_3km.variables["EASE_column_index_3km"][0, 0]
            lon_end = group_3km.variables["EASE_column_index_3km"][0, -1]

            if lat_end <= lat_indices[0] or lat_start >= lat_indices[1] \
                    or lon_end <= lon_indices[0] or lon_start >= lon_indices[1]:
                fh_in.close()
                continue

            print(f)

            lats = global_lats[lat_start: lat_end + 1]
            lons = global_lons[lon_start: lon_end + 1]

            fh_out = Dataset(os.path.join(out_path, f[:-3] + ".nc"), "w")
            fh_out.createDimension("lat", len(lats))
            fh_out.createDimension("lon", len(lons))

            for var_name in flag_dic.values():
                outVar = fh_out.createVariable(var_name, 'u1', ('lat', 'lon',))
                outVar.setncatts({'units': 'NA'})
                outVar.setncatts({'_FillValue': np.array([255]).astype('u1')})
                outVar[:] = ma.array(np.zeros((len(lats), len(lons))),
                                     mask=ma.getmaskarray(group_3km.variables["surface_flag_3km"][:]))

            for v_name, varin in group_3km.variables.items():
                if v_name in ["latitude_3km", "longitude_3km"]:
                    outVar = fh_out.createVariable(v_name[:3], varin.datatype, (v_name[:3]))
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                elif v_name == "soil_moisture_3km":
                    outVar = fh_out.createVariable(v_name[:-4], varin.datatype, ("lat", "lon"))
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    outVar[:] = varin[:]
            fh_out.variables["lat"].setncatts({"lat_start": lat_start})
            fh_out.variables["lat"].setncatts({"lat_end": lat_end})
            fh_out.variables["lon"].setncatts({"lon_start": lon_start})
            fh_out.variables["lon"].setncatts({"lon_end": lon_end})
            fh_out.variables["lat"][:] = lats[:]
            fh_out.variables["lon"][:] = lons[:]

            surface_flag = group_3km.variables["surface_flag_3km"][:]
            surface_flag_mask = ma.getmaskarray(group_3km.variables["surface_flag_3km"][:])
            for i in range(len(lats)):
                for j in range(len(lons)):
                    if not surface_flag_mask[i, j]:
                        bit_sf = '{0:016b}'.format(surface_flag[i, j])[::-1]
                        for bit_index in flag_dic:
                            if bit_sf[bit_index] == "1":
                                fh_out.variables[flag_dic[bit_index]][i, j] = 1

            fh_out.variables["anomalous_sigma0_flag"][:] = ma.array(fh_out.variables["anomalous_sigma0_flag"][:],
                                                                    mask=ma.getmaskarray(
                                                                        group_3km.variables["soil_moisture_3km"][:]))
            fh_out.variables["edge_cell_flag"][:] = ma.array(fh_out.variables["edge_cell_flag"][:],
                                                             mask=ma.getmaskarray(
                                                                 group_3km.variables["soil_moisture_3km"][:]))

            fh_in.close()
            fh_out.close()


def _extract_surface_flags_local(lat1, lat2, lon1, lon2, area_name):
    flag_dic = {0: "static_water_body_flag", 2: "coastal_mask_flag", 3: "urban_area_flag",
                4: "precipitation_flag", 5: "snow_or_ice_flag", 6: "permanent_snow_or_ice_flag",
                7: "frozen_ground_flag", 8: "frozen_ground_st_based",
                9: "mountainous_terrain_flag", 10: "dense_vegetation_flag",
                11: "edge_cell_flag", 12: "anomalous_sigma0_flag"}

    lat_indices, lon_indices = select_area(lat1, lat2, lon1, lon2, "M03")
    lats, lons = get_lat_lon("M03")
    assert (len(lats) != 0 and len(lons) != 0)
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]

    out_path = get_out_path(os.path.join("Data", "Sentinel"))
    fh_out = Dataset(os.path.join(out_path, "static_surface_flags_" + area_name + ".nc"), "w")
    fh_out.createDimension("lat", len(lats))
    fh_out.createDimension("lon", len(lons))

    fh_out.variables["lat"][:] = lats[:]
    fh_out.variables["lon"][:] = lons[:]
    for var in flag_dic.values():
        outVar = fh_out.createVariable(var, 'u1', ('lat', 'lon',))
        outVar.setncatts({'units': 'NA'})
        outVar.setncatts({'_FillValue': np.array([255]).astype('u1')})

    general_in_path = os.path.join("n5eil01u.ecs.nsidc.org", "SMAP", "SPL2SMAP_S.002")
    for doy_folder in os.listdir(general_in_path):
        if doy_folder.startswith("20"):
            for f in os.listdir(os.path.join(general_in_path, doy_folder)):
                if f.endswith(".h5"):
                    fh_in = Dataset(os.path.join(general_in_path, doy_folder, f), "r")
                    group_3km = fh_in.groups["Soil_Moisture_Retrieval_Data_3km"]

                    lat_start = group_3km.variables["EASE_row_index_3km"][0, 0]
                    lat_end = group_3km.variables["EASE_row_index_3km"][-1, 0]
                    lon_start = group_3km.variables["EASE_column_index_3km"][0, 0]
                    lon_end = group_3km.variables["EASE_column_index_3km"][0, -1]

                    if lat_end <= lat_indices[0] or lat_start >= lat_indices[1] \
                            or lon_end <= lon_indices[0] or lon_start >= lon_indices[1]:
                        fh_in.close()
                        continue

                    out_lat_start = max(lat_start - lat_indices[0], 0)
                    out_lat_end = min(lat_end + 1 - lat_indices[0], len(lats))
                    out_lon_start = max(lon_start - lon_indices[0], 0)
                    out_lon_end = min(lon_end + 1 - lon_indices[0], len(lons))
                    in_lat_start = max(lat_indices[0] - lat_start, 0)
                    in_lat_end = min(lat_indices[1] - lat_start, lat_end - lat_start + 1)
                    in_lon_start = max(lon_indices[0] - lon_start, 0)
                    in_lon_end = min(lon_indices[1] - lon_start, lon_end - lon_start + 1)
                    assert (out_lat_end - out_lat_start == in_lat_end - in_lat_start)
                    assert (out_lon_end - out_lon_start == in_lon_end - in_lon_start)
                    for v_name, varin in group_3km.variables.items():
                        if v_name in var_list:
                            a = fh_out.variables[v_name[:-4]][out_lat_start: out_lat_end, out_lon_start: out_lon_end]
                            b = varin[in_lat_start: in_lat_end, in_lon_start: in_lon_end]
                            if not isinstance(a, ma.MaskedArray):
                                a = ma.array(a, mask=np.zeros(a.shape))
                            if not isinstance(b, ma.MaskedArray):
                                b = ma.array(b, mask=np.zeros(b.shape))

                            for i in range(0, out_lat_end - out_lat_start):
                                for j in range(0, out_lon_end - out_lon_start):
                                    if not ma.getmask(a)[i, j] and not ma.getmask(b)[i, j]:
                                        assert a[i, j] == b[i, j], v_name + str(a[i, j]) + str(b[i, j])
                                    elif ma.getmask(a)[i, j] and not ma.getmask(b)[i, j]:
                                        a[i, j] = b[i, j]
                            fh_out.variables[v_name[:-4]][out_lat_start: out_lat_end, out_lon_start: out_lon_end] = a

                    fh_in.close()
    fh_out.close()


def subset_combined_3km(in_path, lat1, lat2, lon1, lon2, area_name, doy, out_path):
    # Used to check the availability of data at specified locations.
    fh_in = Dataset(os.path.join(in_path, doy + ".nc"), "r")

    lat_indices, lon_indices = select_area(lat1, lat2, lon1, lon2, "M03")
    lats, lons = get_lat_lon("M03")
    assert (len(lats) != 0 and len(lons) != 0)
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]

    i_lat_start, i_lat_end, i_lon_start, i_lon_end = match_lat_lon(fh_in.variables["lat"][:],
                                                                   fh_in.variables["lon"][:],
                                                                   lats,
                                                                   lons)

    sub_sm_3km = fh_in.variables["soil_moisture"][i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1]
    if sub_sm_3km.count() > 0:
        print(area_name, doy)
        out_path = get_out_path(os.path.join(out_path, area_name))
        fh_out = Dataset(os.path.join(out_path, doy + ".nc"), "w")

        fh_out.createDimension('lat', len(lats))
        fh_out.createDimension('lon', len(lons))

        outVar = fh_out.createVariable('lat', 'f4', ('lat'))
        outVar.setncatts({"units": "degree_north"})
        outVar[:] = lats[:]
        outVar = fh_out.createVariable('lon', 'f4', ('lon'))
        outVar.setncatts({"units": "degree_east"})
        outVar[:] = lons[:]

        for v_name, varin in fh_in.variables.items():
            if v_name != "lat" and v_name != "lon":
                outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                outVar[:] = varin[i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1]

        fh_out.close()


def convert_rb_to_db(in_path, out_path, doy_start, doy_end, selected_vars):
    out_path = get_out_path(out_path)

    for doy in generate_doy(doy_start, doy_end, ""):
        fh_in = Dataset(os.path.join(in_path, doy + ".nc"), "r")
        fh_out = Dataset(os.path.join(out_path, doy + ".nc"), "w")

        for name, dim in fh_in.dimensions.items():
            fh_out.createDimension(name, len(dim))

        for v_name, varin in fh_in.variables.items():
            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
            if v_name not in selected_vars:
                outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                outVar[:] = varin[:]
            else:
                outVar[:] = 10 * ma.log(varin[:]) / ma.log(10)

        fh_in.close()
        fh_out.close()
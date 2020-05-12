# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from data_preprocessing.rescaling.utils import get_lat_lon_bins
from data_preprocessing.utils import get_lat_lon
from data_preprocessing.utils import select_area

import pandas as pd
from netCDF4 import Dataset
from collections import defaultdict


def fine_scale_convert_to_nc(in_file, out_file, lat_name, lon_name, v_name, insitu):
    fs_values = pd.read_csv(in_file)

    fh_out = Dataset(out_file, "w")
    lats, lons, lat_bins, lon_bins = get_lat_lon_bins("M03", 50, 24, -125, -66)

    fh_out.createDimension("lat", len(lats))
    fh_out.createDimension("lon", len(lons))

    outVar = fh_out.createVariable('lat', float, ('lat'))
    outVar.setncatts({"units": "degree_north"})
    outVar[:] = lats[:]
    outVar = fh_out.createVariable('lon', float, ('lon'))
    outVar.setncatts({"units": "degree_east"})
    outVar[:] = lons[:]

    fs_sm = fh_out.createVariable('fine_scale_soil_moisture', float, ('lat', "lon"))

    for lat, lon, sm_value in zip(fs_values[lat_name], fs_values[lon_name], fs_values[v_name]):
        i_index, j_index = -1, -1
        min_distance = 99999
        for i, lat_v in enumerate(lats):
            if abs(lat - lat_v) < min_distance:
                i_index = i
                min_distance = abs(lat - lat_v)
        assert i_index != -1
        min_distance = 99999
        for j, lon_v in enumerate(lons):
            if abs(lon - lon_v) < min_distance:
                j_index = j
                min_distance = abs(lon - lon_v)
        assert j_index != -1
        if min_distance < 1:
            if insitu == "SCAN":
                fs_sm[i_index, j_index] = sm_value / 100.0
            elif insitu == "USCRN":
                fs_sm[i_index, j_index] = sm_value

    fh_out.close()


def fine_scale_convert_to_nc_average(in_file, out_file, lat_name, lon_name, v_name):
    fs_values = pd.read_csv(in_file)

    fh_out = Dataset(out_file, "w")
    lats, lons, lat_bins, lon_bins = get_lat_lon_bins("M03", 50, 24, -125, -66)

    fh_out.createDimension("lat", len(lats))
    fh_out.createDimension("lon", len(lons))

    outVar = fh_out.createVariable('lat', float, ('lat'))
    outVar.setncatts({"units": "degree_north"})
    outVar[:] = lats[:]
    outVar = fh_out.createVariable('lon', float, ('lon'))
    outVar.setncatts({"units": "degree_east"})
    outVar[:] = lons[:]

    fs_sm = fh_out.createVariable('fine_scale_soil_moisture', float, ('lat', "lon"))
    sm_value_lis = defaultdict(list)
    n_count = 0

    for lat, lon, sm_value in zip(fs_values[lat_name], fs_values[lon_name], fs_values[v_name]):
        i_index, j_index = -1, -1
        min_distance = 99999
        for i, lat_v in enumerate(lats):
            if abs(lat - lat_v) < min_distance:
                i_index = i
                min_distance = abs(lat - lat_v)
        assert i_index != -1
        min_distance = 99999
        for j, lon_v in enumerate(lons):
            if abs(lon - lon_v) < min_distance:
                j_index = j
                min_distance = abs(lon - lon_v)
        assert j_index != -1
        if min_distance < 1:
            sm_value_lis[(i_index, j_index)].append(sm_value)
            n_count += 1

    for key, value in sm_value_lis.items():
        fs_sm[key[0], key[1]] = sum(value) / len(value)

    fh_out.close()


def fine_scale_convert_to_nc_average_with_error(lat1, lat2, lon1, lon2, in_file, out_file, lat_name, lon_name, v_name, v_error_name):
    fs_values = pd.read_csv(in_file)

    fh_out = Dataset(out_file, "w")
    lat_indices, lon_indices = select_area(lat1, lat2, lon1, lon2, "M03")
    lats, lons = get_lat_lon("M03")
    assert (len(lats) != 0 and len(lons) != 0)
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]

    fh_out.createDimension("lat", len(lats))
    fh_out.createDimension("lon", len(lons))

    outVar = fh_out.createVariable('lat', float, ('lat'))
    outVar.setncatts({"units": "degree_north"})
    outVar[:] = lats[:]
    outVar = fh_out.createVariable('lon', float, ('lon'))
    outVar.setncatts({"units": "degree_east"})
    outVar[:] = lons[:]

    fs_sm = fh_out.createVariable('fine_scale_soil_moisture', float, ('lat', "lon"))
    fs_sm_error = fh_out.createVariable('fine_scale_soil_moisture_error', float, ('lat', "lon"))
    sm_value_lis = defaultdict(list)
    sm_error_value_lis = defaultdict(list)
    n_count = 0

    for lat, lon, sm_value, sm_error_value in zip(fs_values[lat_name], fs_values[lon_name], fs_values[v_name],
                                                  fs_values[v_error_name]):
        i_index, j_index = -1, -1
        min_distance = 99999
        for i, lat_v in enumerate(lats):
            if abs(lat - lat_v) < min_distance:
                i_index = i
                min_distance = abs(lat - lat_v)
        assert i_index != -1
        min_distance = 99999
        for j, lon_v in enumerate(lons):
            if abs(lon - lon_v) < min_distance:
                j_index = j
                min_distance = abs(lon - lon_v)
        assert j_index != -1
        if min_distance < 1:
            sm_value_lis[(i_index, j_index)].append(sm_value)
            sm_error_value_lis[(i_index, j_index)].append(sm_error_value)
            n_count += 1

    for key, value in sm_value_lis.items():
        fs_sm[key[0], key[1]] = sum(value) / len(value)
    for key, value in sm_error_value_lis.items():
        fs_sm_error[key[0], key[1]] = sum(value) / len(value)
    fh_out.close()

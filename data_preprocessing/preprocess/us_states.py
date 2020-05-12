# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..utils import select_area, get_lat_lon, match_lat_lon

import os
# import fiona
from collections import defaultdict
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma


def extract_coordinates():
    in_path = os.path.join("Data", "US_States", "shapefiles")
    out_path = os.path.join("Data", "US_States", "coordinates")

    with open(os.path.join("Data", "US_states", "queries_info.txt"), "r") as f:
        start = f.readline().rstrip('\n')
        end = f.readline().rstrip('\n')

    shapefile = fiona.open(os.path.join(in_path, "cb_2016_us_state_20m.shp"))
    shapefile.next()
    for state in shapefile.iterator:
        print(state["properties"]["NAME"])
        with open(os.path.join(out_path, state["properties"]["NAME"] + ".json"), "w") as f:
            f.write(start + str(state["geometry"]["coordinates"]).replace("(", "[").replace(")", "]") + end)


def extract_coornidates_extra():
    in_path = os.path.join("Data", "US_States", "shapefiles")
    out_path = os.path.join("Data", "US_States", "coordinates")

    var_list = ["Florida", "Alaska", "Massachusetts", "Ohio", "Maryland", "Virginia",
                "New York",
                "Washington", "California"]
    var_dic = defaultdict(list)

    with open(os.path.join("Data", "US_states", "queries_info.txt"), "r") as f:
        start = f.readline().rstrip('\n')
        end = f.readline().rstrip('\n')

    shapefile = fiona.open(os.path.join(in_path, "st99_d00.shp"))
    shapefile.next()
    for state in shapefile.iterator:
        if state["properties"]["NAME"] in var_list:
            var_dic[state["properties"]["NAME"]].append(state["geometry"]["coordinates"])

    for v_name in var_dic:
        print(v_name)
        coords = max(var_dic[v_name], key=len)
        with open(os.path.join(out_path, v_name + ".json"), "w") as f:
            f.write(start + str(coords).replace("(", "[").replace(")", "]") + end)


def extract_coornidates_extra_extra():
    in_path = os.path.join("Data", "US_States", "shapefiles")
    out_path = os.path.join("Data", "US_States", "coordinates")

    var_list = ["Wisconsin", "Michigan"]
    var_dic = defaultdict(list)

    with open(os.path.join("Data", "US_states", "queries_info.txt"), "r") as f:
        start = f.readline().rstrip('\n')
        end = f.readline().rstrip('\n')

    shapefile = fiona.open(os.path.join(in_path, "st99_d00.shp"))
    shapefile.next()
    for state in shapefile.iterator:
        if state["properties"]["NAME"] in var_list:
            var_dic[state["properties"]["NAME"]].append(state["geometry"]["coordinates"])

    for v_name in var_dic:
        print(v_name)
        for i_coords, coords in enumerate(var_dic[v_name]):
            with open(os.path.join(out_path, v_name + str(i_coords+1) + ".json"), "w") as f:
                f.write(start + str(coords).replace("(", "[").replace(")", "]") + end)


def combine_to_usa():
    in_path = os.path.join("Data", "US_States", "states", "3km")
    out_path = os.path.join("Data", "US_States")

    fh_out = Dataset(os.path.join(out_path, "usa_states.nc"), "w")

    lat_indices, lon_indices = select_area(50, 24, -125, -66, "M03")
    lats, lons = get_lat_lon("M03")
    assert (len(lats) != 0 and len(lons) != 0)
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]

    fh_out.createDimension("lat", len(lats))
    fh_out.createDimension("lon", len(lons))

    outVar = fh_out.createVariable("states_flag", 'u1', ('lat', 'lon',))
    outVar.setncatts({'_FillValue': np.array([0]).astype('u1')})
    outVar[:] = np.zeros((len(lats), len(lons)))

    index = 1
    index_dic = {}
    first_flag = True
    for nc_file in os.listdir(in_path):
        if nc_file.endswith("nc"):
            fh_state = Dataset(os.path.join(in_path, nc_file), "r")

            if first_flag:
                for v_name, varin in fh_state.variables.items():
                    if v_name in ["lat", "lon"]:
                        outVar = fh_out.createVariable(v_name, varin.datatype, (v_name,))
                        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                fh_out.variables["lat"][:] = lats[:]
                fh_out.variables["lon"][:] = lons[:]
                first_flag = False

            i_lat_start, i_lat_end, i_lon_start, i_lon_end = match_lat_lon(fh_state.variables["lat"][:],
                                                                           fh_state.variables["lon"][:],
                                                                           lats,
                                                                           lons)

            mask_array = ma.getmaskarray(fh_state.variables["elevation"][:])
            for i_out, i_in in zip(range(i_lat_start, i_lat_end + 1), range(len(fh_state.variables["lat"][:]))):
                for j_out, j_in in zip(range(i_lon_start, i_lon_end + 1), range(len(fh_state.variables["lon"][:]))):
                    if not mask_array[i_in, j_in]:
                        fh_out.variables["states_flag"][i_out, j_out] = index
            index_dic[nc_file[:-3]] = index
            index += 1

            fh_state.close()
    print(index_dic)
    fh_out.close()


def flag_last_two_states():
    fh_sand = Dataset(os.path.join("Data", "SoilType", "3km", "soiltype_usa.nc"), "r")
    sand_mask = ma.getmaskarray(fh_sand.variables["sand"][:])
    lats, lons = fh_sand.variables["lat"][:], fh_sand.variables["lon"][:]

    fh_usa_states = Dataset(os.path.join("Data", "US_States", "usa_states_missing_last_two.nc"), "r")
    states_values = fh_usa_states.variables["states_flag"][:].filled()

    fh_out = Dataset(os.path.join("Data", "US_States", "usa_states.nc"), "w")
    fh_out.createDimension("lat", len(lats))
    fh_out.createDimension("lon", len(lons))
    for v_name, varin in fh_usa_states.variables.items():
        if v_name in ["lat", "lon"]:
            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[:]

    for i in range(len(lats)):
        for j in range(len(lons)):
            if not sand_mask[i,j] and lons[j] < -80.5 and states_values[i,j] == 0.0:
                states_values[i,j] = 20
            elif not sand_mask[i, j] and lons[j] > -80.5 and states_values[i, j] == 0:
                states_values[i, j] = 48

    outVar = fh_out.createVariable("states_flag", 'u1', ('lat', 'lon',))
    outVar.setncatts({'_FillValue': np.array([0]).astype('u1')})
    outVar[:] = ma.array(states_values, mask=sand_mask)

    fh_sand.close()
    fh_usa_states.close()
    fh_out.close()








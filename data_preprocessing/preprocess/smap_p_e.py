# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause
from ..utils import get_out_path, get_lat_lon, select_area

import os
from netCDF4 import Dataset


def convert_to_nc(doy):
    ignore_vars = ["EASE_column_index", "EASE_row_index", "latitude_centroid", "longitude_centroid", "tb_time_seconds",
                   "tb_time_utc"]

    in_path = os.path.join("n5eil01u.ecs.nsidc.org", "SMAP", "SPL3SMP_E.001")
    out_path = get_out_path(os.path.join("Data", "SMAP_P_E", "global"))

    fh_out = Dataset(os.path.join(out_path, doy + ".nc"), "w")
    fh_out.createDimension("lat", 1624)
    fh_out.createDimension("lon", 3856)
    lats, lons = get_lat_lon("M09")
    assert (len(lats) != 0 and len(lons) != 0)

    for f in os.listdir(in_path):
        if doy in f:
            print(f)
            fh_in = Dataset(os.path.join(in_path, f), "r")
            group = fh_in.groups["Soil_Moisture_Retrieval_Data_AM"]

            for v_name, varin in group.variables.items():
                if v_name in ["latitude", "longitude"]:
                    outVar = fh_out.createVariable(v_name[:3], varin.datatype, (v_name[:3]))
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                elif v_name not in ignore_vars:
                    outVar = fh_out.createVariable(v_name, varin.datatype, ("lat", "lon"))
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    outVar[:] = varin[:]

            fh_out.variables["lat"][:] = lats[:]
            fh_out.variables["lon"][:] = lons[:]

            fh_in.close()
    fh_out.close()


def convert_to_nc_local(doy, lat1, lat2, lon1, lon2, area_name):
    lat_indices, lon_indices = select_area(lat1, lat2, lon1, lon2, "M09")
    lats, lons = get_lat_lon("M09")
    assert (len(lats) != 0 and len(lons) != 0)
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]

    ignore_vars = ["EASE_column_index", "EASE_row_index", "latitude_centroid", "longitude_centroid", "tb_time_seconds",
                   "tb_time_utc"]

    in_path = os.path.join("n5eil01u.ecs.nsidc.org", "SMAP", "SPL3SMP_E.002")
    out_path = get_out_path(os.path.join("Data", "SMAP_P_E", area_name))

    fh_out = Dataset(os.path.join(out_path, doy + ".nc"), "w")
    fh_out.createDimension("lat", len(lats))
    fh_out.createDimension("lon", len(lons))

    for f in os.listdir(in_path):
        if doy in f:
            print(f)
            fh_in = Dataset(os.path.join(in_path, f), "r")
            group = fh_in.groups["Soil_Moisture_Retrieval_Data_AM"]

            for v_name, varin in group.variables.items():
                if v_name in ["latitude", "longitude"]:
                    outVar = fh_out.createVariable(v_name[:3], varin.datatype, (v_name[:3]))
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                elif v_name not in ignore_vars:
                    outVar = fh_out.createVariable(v_name, varin.datatype, ("lat", "lon"))
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    outVar[:] = varin[lat_indices[0]: lat_indices[1], lon_indices[0]: lon_indices[1]]

            fh_out.variables["lat"][:] = lats[:]
            fh_out.variables["lon"][:] = lons[:]

            fh_in.close()
    fh_out.close()

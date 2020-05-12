# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from . import select_area, get_lat_lon, match_lat_lon

from netCDF4 import Dataset


def subset_usa(in_file, lat1, lat2, lon1, lon2, var_lis, out_file):
    fh_in = Dataset(in_file, "r")
    fh_out = Dataset(out_file, "w")

    lat_indices, lon_indices = select_area(lat1, lat2, lon1, lon2, "M03")
    lats, lons = get_lat_lon("M03")
    lats = lats[lat_indices[0]: lat_indices[1]]
    lons = lons[lon_indices[0]: lon_indices[1]]

    i_lat_start, i_lat_end, i_lon_start, i_lon_end = match_lat_lon(fh_in.variables["lat"][:],
                                                                   fh_in.variables["lon"][:],
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

    for v_name, varin in fh_in.variables.items():
        if v_name in var_lis:
            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[i_lat_start: i_lat_end + 1, i_lon_start: i_lon_end + 1]

    fh_out.close()


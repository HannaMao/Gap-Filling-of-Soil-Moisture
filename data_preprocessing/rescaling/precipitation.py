# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from .bilinear import Kdtree_fast, bilinear
from .utils import get_lat_lon_bins
from ..utils import get_out_path, timeit, timenow

import os
from netCDF4 import Dataset
from datetime import datetime
import numpy as np


def gpm_downsample(doy_start, doy_end, lat1, lat2, lon1, lon2):
    in_path = os.path.join("Data", "GPM", "0.1degree")
    out_path = get_out_path(os.path.join("Data", "GPM", "3km"))

    var_lis = ["precipitation"]

    date_start = datetime.strptime(doy_start, "%Y%m%d").date()
    date_end = datetime.strptime(doy_end, "%Y%m%d").date()

    lats, lons, _, _ = get_lat_lon_bins("M03", lat1, lat2, lon1, lon2)

    for nc_file in os.listdir(in_path):
        if nc_file.endswith('.nc'):
            nc_date = datetime.strptime(nc_file[:-3], "%Y%m%d").date()
            if date_start <= nc_date <= date_end:
                print(nc_file)
                fh_in = Dataset(os.path.join(in_path, nc_file), mode='r')
                fh_out = Dataset(os.path.join(out_path, nc_file), 'w')

                timeit()
                ns = Kdtree_fast(fh_in, 'lat', 'lon')
                dic_var = {}
                for var in var_lis:
                    dic_var[var] = fh_in.variables[var][:]
                timenow()

                fh_out.createDimension('lat', len(lats))
                fh_out.createDimension('lon', len(lons))

                outVar = fh_out.createVariable('lat', 'f4', ('lat',))
                outVar.setncatts({"units": "degree_north"})
                outVar[:] = lats[:]
                outVar = fh_out.createVariable('lon', 'f4', ('lon',))
                outVar.setncatts({"units": "degree_east"})
                outVar[:] = lons[:]

                for v_name, varin in fh_in.variables.items():
                    if v_name in var_lis:
                        outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})

                for idx, lat_var in enumerate(lats):
                    for idy, lon_var in enumerate(lons):
                        dist_sq, iy, ix = ns.query(lat_var, lon_var)
                        for var in var_lis:
                            fh_out.variables[var][idx, idy] = \
                                bilinear(lat_var, lon_var, iy, ix, ns.latvar, ns.lonvar, dic_var[var], mask_value=None)
                fh_in.close()
                fh_out.close()


def gpm_downsample_given_doys(doy_list, lat1, lat2, lon1, lon2):
    in_path = os.path.join("Data", "GPM", "0.1degree")
    out_path = get_out_path(os.path.join("Data", "GPM", "3km"))

    var_lis = ["precipitation"]

    lats, lons, _, _ = get_lat_lon_bins("M03", lat1, lat2, lon1, lon2)
    first_flag = True
    ns = None
    dic_var = {}

    for doy in doy_list:
        nc_file = doy + ".nc"
        if nc_file in os.listdir(in_path):
            print(nc_file)
            fh_in = Dataset(os.path.join(in_path, nc_file), mode='r')
            fh_out = Dataset(os.path.join(out_path, nc_file), 'w')

            if first_flag:
                timeit()
                ns = Kdtree_fast(fh_in, 'lat', 'lon')
                for var in var_lis:
                    dic_var[var] = fh_in.variables[var][:]
                first_flag = False
                timenow()

            fh_out.createDimension('lat', len(lats))
            fh_out.createDimension('lon', len(lons))

            outVar = fh_out.createVariable('lat', 'f4', ('lat',))
            outVar.setncatts({"units": "degree_north"})
            outVar[:] = lats[:]
            outVar = fh_out.createVariable('lon', 'f4', ('lon',))
            outVar.setncatts({"units": "degree_east"})
            outVar[:] = lons[:]

            for v_name, varin in fh_in.variables.items():
                if v_name in var_lis:
                    outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})

            for idx, lat_var in enumerate(lats):
                for idy, lon_var in enumerate(lons):
                    dist_sq, iy, ix = ns.query(lat_var, lon_var)
                    for var in var_lis:
                        fh_out.variables[var][idx, idy] = \
                            bilinear(lat_var, lon_var, iy, ix, ns.latvar, ns.lonvar, dic_var[var], mask_value=None)
            fh_in.close()
            fh_out.close()


def find_nearest_index(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def get_mapped_indices(lats, lons, origi_lats, origi_lons):
    mapped_indices = []
    for lat_value in lats:
        for lon_value in lons:
            mapped_indices.append((find_nearest_index(origi_lats, lat_value),
                                   find_nearest_index(origi_lons, lon_value)))
    return mapped_indices


def gpm_downsample_nn_given_doys(doy_list, lat1, lat2, lon1, lon2):
    in_path = os.path.join("Data", "GPM", "0.1degree")
    out_path = get_out_path(os.path.join("Data", "GPM", "3km"))

    var_lis = ["precipitation"]

    lats, lons, _, _ = get_lat_lon_bins("M03", lat1, lat2, lon1, lon2)

    first_flag = True
    mapped_indices = None

    for doy in doy_list:
        nc_file = doy + ".nc"
        if nc_file in os.listdir(in_path):
            print(nc_file)
            fh_in = Dataset(os.path.join(in_path, nc_file), mode='r')
            fh_out = Dataset(os.path.join(out_path, nc_file), 'w')

            if first_flag:
                origi_lats, origi_lons = fh_in.variables['lat'][:], fh_in.variables['lon'][:]
                mapped_indices = get_mapped_indices(lats, lons, origi_lats, origi_lons)
                first_flag = False

            fh_out.createDimension('lat', len(lats))
            fh_out.createDimension('lon', len(lons))

            outVar = fh_out.createVariable('lat', 'f4', ('lat',))
            outVar.setncatts({"units": "degree_north"})
            outVar[:] = lats[:]
            outVar = fh_out.createVariable('lon', 'f4', ('lon',))
            outVar.setncatts({"units": "degree_east"})
            outVar[:] = lons[:]

            for v_name, varin in fh_in.variables.items():
                if v_name in var_lis:
                    outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    n = 0
                    for idx, lat_var in enumerate(lats):
                        for idy, lon_var in enumerate(lons):
                            origi_idx, origi_idy = mapped_indices[n]
                            fh_out.variables[v_name][idx, idy] = varin[origi_idx, origi_idy]
                            n += 1
            fh_in.close()
            fh_out.close()

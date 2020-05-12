# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause
import os
import pandas as pd
import numpy.ma as ma
from netCDF4 import Dataset

from .utils import find_index, get_lat_lon


def out2NC(path, f_reference, fName):
    df = pd.read_csv(os.path.join(path, fName))
    var_list = list(df.columns.values)

    fh_reference = Dataset(f_reference, "r")
    fh = Dataset(os.path.join(path, (os.path.splitext(fName)[0]) + ".nc"), "w")

    for name, dim in fh_reference.dimensions.items():
        fh.createDimension(name, len(dim))

    for v_name, varin in fh_reference.variables.items():
        if v_name in var_list:
            outVar = fh.createVariable(v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            if v_name in ["lat", "lon"]:
                outVar[:] = varin[:]

    lat = fh_reference.variables['lat'][:]
    lon = fh_reference.variables['lon'][:]

    for index, row in df.iterrows():
        lat_index = find_index(lat, row["lat"])
        lon_index = find_index(lon, row["lon"])
        for var in list(set(var_list) - {"lat", "lon"}):
            fh.variables[var][lat_index, lon_index] = row[var]


def convert2nc(path, f_reference, fName=None):
    if fName:
        out2NC(path, f_reference, fName + ".csv")
    else:
        for csv_file in os.listdir(path):
            if csv_file.endswith('.csv'):
                out2NC(path, f_reference, csv_file)

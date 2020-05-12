# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..utils import get_out_path

import os
from datetime import datetime
from netCDF4 import Dataset
import numpy as np


def gpm_extract(folder, doy_start, doy_end):
    in_path = os.path.join("n5eil01u.ecs.nsidc.org", "GPM_L3", folder)
    out_path = get_out_path(os.path.join("Data", "GPM", "0.1degree"))

    date_start = datetime.strptime(doy_start, "%Y%m%d").date()
    date_end = datetime.strptime(doy_end, "%Y%m%d").date()

    for nc_file in os.listdir(in_path):
        if nc_file.endswith(".nc4"):
            nc_doy = nc_file.split(".")[4].split("-")[0]
            nc_date = datetime.strptime(nc_doy, "%Y%m%d").date()
            if date_start <= nc_date <= date_end:
                print(nc_file)
                fh_in = Dataset(os.path.join(in_path, nc_file), "r")
                fh_out = Dataset(os.path.join(out_path, nc_doy + ".nc"), "w")

                for name, dim in fh_in.dimensions.items():
                    fh_out.createDimension(name, len(dim))

                for v_name, varin in fh_in.variables.items():
                    if v_name == "lat":
                        outVar = fh_out.createVariable(v_name, varin.datatype, "lat")
                        outVar.setncatts({"units": "degree_north"})
                        outVar[:] = varin[::-1]
                    elif v_name == "lon":
                        outVar = fh_out.createVariable(v_name, varin.datatype, "lon")
                        outVar.setncatts({"units": "degree_east"})
                        outVar[:] = varin[:]
                    elif v_name == "precipitationCal" or v_name == "randomError":
                        new_name = "precipitation" if v_name == "precipitationCal" else "random_error"
                        outVar = fh_out.createVariable(new_name, varin.datatype, ("lat", "lon"))
                        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                        outVar[:] = np.flipud(varin[:].transpose())

                fh_in.close()
                fh_out.close()



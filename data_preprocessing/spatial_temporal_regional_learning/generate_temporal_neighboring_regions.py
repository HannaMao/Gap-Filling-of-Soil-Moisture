# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from ..utils import get_out_path

from netCDF4 import Dataset
import numpy.ma as ma
import os
from datetime import datetime


def generate_temporal_neighboring_regions(in_file, search_path, out_folder, variable, mode, n_hist=None, n_cohort=None):
    fh_in = Dataset(in_file, "r")

    in_doy = datetime.strptime(in_file.split("/")[-1][:-3], "%Y%m%d").date()
    in_mask = ma.getmaskarray(fh_in.variables[variable][:])
    out_folder = get_out_path(out_folder)

    temp_candidates = [nc_file for nc_file in os.listdir(search_path) if nc_file.endswith(".nc")]
    temp_candidates = sorted(temp_candidates, key=lambda x: datetime.strptime(x[:-3], '%Y%m%d'))[::-1]
    for nc_file in temp_candidates:
        nc_doy = datetime.strptime(nc_file[:-3], "%Y%m%d").date()
        doy_diff = (in_doy - nc_doy).days
        if (mode == "window" and (0 <= doy_diff <= n_hist)) \
                or (mode == "most_recent" and doy_diff > 0) \
                or (mode == "cohort" and (doy_diff // 12 == (n_cohort - 1))):
            fh_doy = Dataset(os.path.join(search_path, nc_file), "r")

            doy_mask = ma.mask_or(ma.getmaskarray(fh_doy.variables[variable][:]), in_mask)

            if ma.any(~doy_mask):
                print(in_file, "--", nc_file, doy_diff)
                fh_out = Dataset(os.path.join(out_folder, nc_file), "w")

                for name, dim in fh_doy.dimensions.items():
                    fh_out.createDimension(name, len(dim))

                for v_name, varin in fh_doy.variables.items():
                    if v_name == 'lat' or v_name == 'lon':
                        outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                        outVar[:] = varin[:]
                    else:
                        outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                        outVar[:] = ma.array(varin[:], mask=doy_mask)

                fh_out.close()
                if mode == "most_recent":
                    fh_doy.close()
                    break
            fh_doy.close()
    fh_in.close()



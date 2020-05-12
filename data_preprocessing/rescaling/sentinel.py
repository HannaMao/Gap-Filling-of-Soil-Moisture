# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from data_preprocessing.utils import get_out_path, generate_doy

import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma


def smap_sentinel_upscale(doy_start, doy_end, selected_vars, out_folder, output_all):
    """
    smap_ap usa 3km index range: lat: [566:1444] lon: [1767:3662], starting from 0, included
    """
    in_path = os.path.join("Data", "Sentinel", "usa")
    out_path = get_out_path(os.path.join("Data", "Sentinel", out_folder))

    for doy in generate_doy(doy_start, doy_end, ""):
        print(doy)
        if doy + ".nc" in os.listdir(in_path):
            fh_in = Dataset(os.path.join(in_path, doy + ".nc"), mode='r')
            fh_out = Dataset(os.path.join(out_path, doy + ".nc"), 'w')

            for name, dim in fh_in.dimensions.items():
                fh_out.createDimension(name, len(dim))

            for v_name, varin in fh_in.variables.items():
                if output_all or v_name in ["lat", "lon"]:
                    outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    outVar[:] = varin[:]

                if v_name in selected_vars:
                    outVar = fh_out.createVariable(v_name + "_9km_mean", varin.datatype, varin.dimensions)
                    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                    varin_value = varin[1:-2, :]
                    n_lat, n_lon = varin_value.shape
                    aggregated_value = np.zeros((n_lat//3, n_lon//3))
                    for i in range(n_lat//3):
                        for j in range(n_lon//3):
                            aggregated_value[i, j] = ma.mean(varin_value[i*3:i*3+3,j*3:j*3+3])
                    aggregated_value = np.repeat(aggregated_value, 3, axis=0)
                    aggregated_value = np.repeat(aggregated_value, 3, axis=1)
                    outVar[:] = ma.masked_invalid(np.vstack((varin_value[:1, :], aggregated_value, varin_value[-2:, :])))

            fh_in.close()
            fh_out.close()
        else:
            if output_all:
                fh_in = Dataset(os.path.join(in_path, "20150401.nc"), mode='r')
                fh_out = Dataset(os.path.join(out_path, doy + ".nc"), 'w')

                for name, dim in fh_in.dimensions.items():
                    fh_out.createDimension(name, len(dim))
                for v_name, varin in fh_in.variables.items():
                    if v_name in ["lat", "lon"]:
                        outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                        outVar[:] = varin[:]
                    else:
                        outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
                        outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
                fh_in.close()
                fh_out.close()




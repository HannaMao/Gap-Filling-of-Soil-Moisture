# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import os
from netCDF4 import Dataset
import numpy.ma as ma


def mask_by_mask_array(mask_array, file_in, file_out, reverse=False):
    if reverse:
        mask_array = ~mask_array
    fh_in = Dataset(file_in, "r")
    fh_out = Dataset(file_out, "w")

    for name, dim in fh_in.dimensions.items():
        fh_out.createDimension(name, len(dim))

    for v_name, varin in fh_in.variables.items():
        if v_name == 'lat' or v_name == 'lon':
            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[:]
        else:
            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            origi_mask = ma.getmaskarray(varin[:])
            outVar[:] = ma.array(varin[:], mask=ma.mask_or(origi_mask, mask_array))

    fh_out.close()
    fh_in.close()


def mask_by_state(state_index, file_in, file_out):
    fh_states = Dataset(os.path.join("Data", "US_States", "usa_states.nc"), "r")
    states_array = fh_states.variables["states_flag"][:]
    mask_array = ma.getmaskarray(ma.masked_where(states_array != state_index, states_array))

    fh_in = Dataset(file_in, "r")
    fh_out = Dataset(file_out, "w")

    for name, dim in fh_in.dimensions.items():
        fh_out.createDimension(name, len(dim))

    for v_name, varin in fh_in.variables.items():
        if v_name == 'lat' or v_name == 'lon':
            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[:]
        else:
            outVar = fh_out.createVariable(v_name, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            origi_mask = ma.getmaskarray(varin[:])
            masked_values = ma.array(varin[:], mask=ma.mask_or(origi_mask, mask_array))
            outVar[:] = masked_values
            if v_name == "soil_moisture":
                if masked_values.count() < 2000:
                    os.remove(file_out)
                    return file_out[-11:-3]

    fh_out.close()
    fh_in.close()

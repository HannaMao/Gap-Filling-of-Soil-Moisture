# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from data_preprocessing import landcover_class_dic
from .utils import get_lat_lon_bins
from data_preprocessing.utils import get_out_path

from netCDF4 import Dataset
import os
import numpy as np
import numpy.ma as ma


def landcover_upsample(lat1, lat2, lon1, lon2, reso, area_name):
    fh_in = Dataset(os.path.join("n5eil01u.ecs.nsidc.org", "MCD12Q1", "15_12_31-16_12_31",
                                 "MCD12Q1.006_500m_aid0001.nc"), "r")
    fh_out = Dataset(os.path.join(get_out_path(os.path.join("Data", "LANDCOVER")),
                                  "landcover_" + reso + "_" + area_name + "_2016.nc"), "w")

    n_dim = str(int(reso[:-2])).zfill(2)
    lats, lons, lat_bins, lon_bins = get_lat_lon_bins("M" + n_dim, lat1, lat2, lon1, lon2)
    fill_value = -9999.0

    dic_var = {}
    for var in ['lat', 'lon']:
        dic_var[var] = fh_in.variables[var]
    dic_var['lat_value'] = dic_var['lat'][::-1]
    dic_var['lon_value'] = dic_var['lon'][:]

    fh_out.createDimension('lat', len(lats))
    fh_out.createDimension('lon', len(lons))

    for var in ['lat', 'lon']:
        outVar = fh_out.createVariable(var, 'f4', (var,))
        outVar.setncatts({k: dic_var[var].getncattr(k) for k in dic_var[var].ncattrs()})
        outVar[:] = lats if var == "lat" else lons

    lc_500m = fh_in.variables["LC_Type1"][1, :, :]
    lc_resampled_dic = {}
    for v in landcover_class_dic.values():
        lc_resampled_dic[v] = np.full((len(lats), len(lons)), 0.0)
    for s in ["1", "2", "3"]:
        lc_resampled_dic["lc_" + s] = np.full((len(lats), len(lons)), fill_value)
        lc_resampled_dic["lc_fraction_" + s] = np.full((len(lats), len(lons)), fill_value)

    for id_lats in range(len(lats)):
        for id_lons in range(len(lons)):
            lats_index = np.searchsorted(dic_var['lat_value'],
                                         [lat_bins[id_lats + 1], lat_bins[id_lats]])
            lons_index = np.searchsorted(dic_var['lon_value'],
                                         [lon_bins[id_lons], lon_bins[id_lons + 1]])
            if lats_index[0] != lats_index[1] and lons_index[0] != lons_index[1]:
                selected = lc_500m[np.array(range(-lats_index[1], -lats_index[0]))[:, None],
                                   np.array(range(lons_index[0], lons_index[1]))]
                selected_size = selected.shape[0] * selected.shape[1]
                selected_compressed = selected.compressed()
                lc_id, lc_count = np.unique(selected_compressed, return_counts=True)
                for i, c in zip(lc_id, lc_count):
                    lc_resampled_dic[landcover_class_dic[i]][id_lats, id_lons] = c / selected_size
                lc_count_sort_ind = np.argsort(-lc_count)
                for i in range(3):
                    if len(lc_id) > i:
                        lc_resampled_dic["lc_" + str(i+1)][id_lats, id_lons] = \
                            lc_id[lc_count_sort_ind[i]]
                        lc_resampled_dic["lc_fraction_" + str(i+1)][id_lats, id_lons] = \
                            lc_count[lc_count_sort_ind[i]] / selected_size
                    else:
                        lc_resampled_dic["lc_" + str(i + 1)][id_lats, id_lons] = fill_value
                        lc_resampled_dic["lc_fraction_" + str(i + 1)][id_lats, id_lons] = fill_value

    for v in landcover_class_dic.values():
        outVar = fh_out.createVariable("lc_" + v, 'f4', ('lat', 'lon',))
        outVar.setncatts({'_FillValue': np.array([-9999.0]).astype('f')})
        outVar[:] = lc_resampled_dic[v][:]
        outVar[:] = ma.masked_less(outVar, -1)
    for s in ["1", "2", "3"]:
        for t in ["lc_", "lc_fraction_"]:
            outVar = fh_out.createVariable(t + s, 'f4', ('lat', 'lon',))
            outVar.setncatts({'_FillValue': np.array([-9999.0]).astype('f')})
            outVar[:] = lc_resampled_dic[t+s][:]
            outVar[:] = ma.masked_less(outVar, -1)

    fh_in.close()
    fh_out.close()

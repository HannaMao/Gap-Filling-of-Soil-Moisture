# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import os
import csv
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
import operator
from datetime import datetime


def _out2csv(path, fheader, f_name, ignore_fields, doy=None):
    ignore_fields = ignore_fields if ignore_fields else []
    with open(os.path.join(path, f_name + '.csv'), "w") as csvfile:
        c = csv.writer(csvfile, delimiter=',')
        if doy is None:
            csv_header = sorted(list(set(fheader.variables.keys()) - set(ignore_fields)))
            insert_indices = {"lat": csv_header.index("lat"),
                              "lon": csv_header.index("lon")}
        else:
            doy = datetime.strptime(doy, '%Y%m%d').strftime('%Y%j')
            csv_header = sorted(list(set(fheader.variables.keys()) - set(ignore_fields)) + ["doy"])
            insert_indices = {"doy": csv_header.index("doy"),
                              "lat": csv_header.index("lat"),
                              "lon": csv_header.index("lon")}
        c.writerow(csv_header)

        var_matrix = []
        set_mask = False
        mask_indices = np.array([])
        length_dic = []
        for var in csv_header:
            if var not in ['doy', 'lat', 'lon'] and var not in ignore_fields:
                if not set_mask:
                    # we will have lon as the row and lat as the column for kfold split vertically
                    mask_indices = ma.where(~ma.getmaskarray(ma.transpose(fheader.variables[var][:])))
                    set_mask = True
                compressed = ma.transpose(fheader.variables[var][:]).compressed()
                length_dic.append(len(compressed))
                var_matrix.append(compressed)

        assert all(length_dic[0] == length for length in length_dic), \
            "Should always be called after all variables are masked to be overlapped"

        dim_indices = {"lon": mask_indices[0], "lat": mask_indices[1]}   # order of indices changes correspondingly
        for dim_name in [_[0] for _ in sorted(insert_indices.items(), key=operator.itemgetter(1))]:
            if dim_name == "lat" or dim_name == "lon":
                var_matrix.insert(insert_indices[dim_name],
                                  [fheader.variables[dim_name][i] for i in dim_indices[dim_name]])
            else:
                var_matrix.insert(insert_indices[dim_name],
                                  [doy] * len(var_matrix[0]))

        num = 0
        for row in zip(*var_matrix):
            c.writerow(row)
            num += 1
        assert(num == len(var_matrix[0]))


def convert2csv(path, f_name=None, ignore_fields=None, doy=None):
    if f_name:
        fh = Dataset(os.path.join(path, f_name + ".nc"), 'r')
        _out2csv(path, fh, f_name, ignore_fields, doy)
    else:
        for nc_file in os.listdir(path):
            if nc_file.endswith('.nc'):
                fh = Dataset(os.path.join(path, nc_file), 'r')
                _out2csv(path, fh, nc_file[:-3], ignore_fields)








# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import os
import csv
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset


def _out2csv_strip_scanner(path, fheader, f_name, ignore_fields):
    ignore_fields = ignore_fields if ignore_fields else []
    with open(os.path.join(path, f_name + '.csv'), "w") as csvfile:
        c = csv.writer(csvfile, delimiter=',')
        csv_header = sorted(list(set(fheader.variables.keys()) - set(ignore_fields)))
        c.writerow(csv_header)

        n_cells = len(fheader.variables["soil_moisture"][:].compressed())
        sm_mask = ma.getmaskarray(fheader.variables["soil_moisture"][:])
        lats, lons = fheader.variables["lat"][:], fheader.variables["lon"][:]
        n_lat, n_lon = len(lats), len(lons)

        while n_cells > 0:
            for i in range(n_lat):
                for j in range(n_lon):
                    if not sm_mask[i, j]:
                        sm_mask[i, j] = True
                        n_cells -= 1
                        out_row = []
                        for v_name in csv_header:
                            if v_name != "lat" and v_name != "lon":
                                out_row.append(fheader.variables[v_name][i, j])
                            elif v_name == "lat":
                                out_row.append(fheader.variables["lat"][i])
                            elif v_name == "lon":
                                out_row.append(fheader.variables["lon"][j])
                        c.writerow(out_row)
                        break
                if n_cells == 0:
                    break

    assert np.all(sm_mask)


def convert2csv_strip_scanner(path, f_name=None, ignore_fields=None):
    if f_name:
        fh = Dataset(os.path.join(path, f_name + ".nc"), 'r')
        _out2csv_strip_scanner(path, fh, f_name, ignore_fields)
    else:
        for nc_file in os.listdir(path):
            if nc_file.endswith('.nc'):
                fh = Dataset(os.path.join(path, nc_file), 'r')
                _out2csv_strip_scanner(path, fh, nc_file[:-3], ignore_fields)










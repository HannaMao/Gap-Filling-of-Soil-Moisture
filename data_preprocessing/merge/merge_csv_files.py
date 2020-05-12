# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import shutil
import os


# https://stackoverflow.com/questions/44791212/concatenating-multiple-csv-files-into-a-single-csv-with-the-same-header-python
def merge_csv_files(in_path, out_path, out_file, in_file_lis=None):
    if not in_file_lis:
        in_file_lis = [csv_file for csv_file in os.listdir(in_path) if csv_file.endswith(".csv")]
    i = 0
    with open(os.path.join(out_path, out_file), "wb") as outfile:
        for csv_file in in_file_lis:
            print(csv_file)
            with open(os.path.join(in_path, csv_file), "rb") as infile:
                if i != 0:
                    infile.readline()
                shutil.copyfileobj(infile, outfile)
                i += 1

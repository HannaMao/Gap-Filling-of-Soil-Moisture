# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import os

from .timing import timeit, timenow
from .get_lat_lon import get_lat_lon, match_lat_lon
from .generate_doy import generate_doy, generate_nearest_doys, generate_most_recent_doys
from .select_area import select_area
from .find_index import find_index
from .logger import Logger
from .float_precision import round_sigfigs
from .concat_csv_files import concat_csv_files
from .average_various_days import average_various_days
from .merge_predict_with_origi import merge_predict_with_orig
from .subset_usa import subset_usa

__all__ = ["get_out_path",
           "timeit", "timenow",
           "get_lat_lon", "match_lat_lon",
           "generate_doy", "generate_nearest_doys", "generate_most_recent_doys",
           "select_area",
           "find_index",
           "Logger",
           "round_sigfigs",
           "concat_csv_files",
           "average_various_days",
           "subset_usa"]

states_index_dic = {"alabama": 1, "arizona": 2, "arkansas": 3, "california": 4, "colorado": 5, "connecticut": 6,
                    "dc": 7, "delaware": 8, "florida": 9, "georgia": 10, "idaho": 11, "illinois": 12, "indiana": 13,
                    "iowa": 14, "kansas": 15, "kentucky": 16, "louisiana": 17, "maryland": 18, "massachusetts": 19,
                    "michigan": 20, "minnesota": 21, "mississippi": 22, "missouri": 23, "montana": 24, "nebraska": 25,
                    "nevada": 26, "new_hampshire": 27, "new_jersey": 28, "new_mexico": 29, "new_york": 30,
                    "north_carolina": 31, "north_dakota": 32, "ohio": 33, "oklahoma": 34, "oregon": 35,
                    "pennsylvania": 36, "south_carolina": 37, "south_dakota": 38, "tennessee": 39, "texas": 40,
                    "utah": 41, "vermont": 42, "virginia": 43, "washington": 44, "west_virginia": 45,
                    "wisconsin": 46, "wyoming": 47}

index_states_dic = {1: "alabama", 2: "arizona", 3: "arkansas", 4: "california", 5: "colorado", 6: "connecticut",
                    7: "dc", 8: "delaware", 9: "florida", 10: "georgia", 11: "idaho", 12: "illinois", 13: "indiana",
                    14: "iowa", 15: "kansas", 16: "kentucky", 17: "louisiana", 18: "maryland", 19: "massachusetts",
                    20: "michigan", 21: "minnesota", 22: "mississippi", 23: "missouri", 24: "montana", 25: "nebraska",
                    26: "nevada", 27: "new_hampshire", 28: "new_jersey", 29: "new_mexico", 30: "new_york",
                    31: "north_carolina", 32: "north_dakota", 33: "ohio", 34: "oklahoma", 35: "oregon",
                    36: "pennsylvania", 37: "south_carolina", 38: "south_dakota", 39: "tennessee", 40: "texas",
                    41: "utah", 42: "vermont", 43: "virginia", 44: "washington", 45: "west_virginia",
                    46: "wisconsin", 47: "wyoming"}


def get_out_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def vprint(output, verbose):
    if verbose:
        print(output)

# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from .csv_to_nc import convert2nc
from .nc_to_csv import convert2csv

__all__ = ["convert2nc", "convert2csv",
           "area_filter", "real_gap_area_filter",
           "area_dic",
           "states_index_dic", "index_states_dic"]

area_filter = {"arizona": {"spatial": [("1AIWDV", "110W36N"), ("1AIWDV", "111W35N"), ("1AIWDV", "111W33N"),
                                       ("1AIWDV", "111W32N"), ("1AIWDV", "112W30N")],
                           "temporal": [("1AIWDV", "113W36N"), ("1AIWDV", "113W34N"), ("1AIWDV", "113W33N"),
                                        ("1AIWDV", "114W31N")]},
               "oklahoma": {"spatial": [("1AIWDV", "097W35N"), ("1AIWDV", "097W36N")],
                            "temporal": [("1AIWDV", "099W37N"), ("1AIWDV", "099W34N"), ("1AIWDV", "099W36N"),
                                         ("1AIWDV", "098W33N"), ("1AIWDV", "099W35N"), ("1AIWDV", "098W34N")]},
               "iowa": {"spatial": [("1AIWDV", "092W43N"), ("1AIWDV", "092W42N"), ("1AIWDV", "092W41N"),
                                    ("1AIWDV", "092W40N")],
                        "temporal": [("1AIWDV", "094W43N"), ("1AIWDV", "094W42N"), ("1AIWDV", "094W41N"),
                                     ("1AIWDV", "094W40N"), ("1BIWDV", "092W43N")]},
               "arkansas": {"spatial": [("1AIWDV", "092W32N"), ("1AIWDV", "092W33N"), ("1AIWDV", "092W34N"),
                                        ("1AIWDV", "093W35N"), ("1AIWDV", "093W36N"), ("1AIWDV", "093W37N")],
                            "temporal": [("1AIWDV", "090W32N"), ("1AIWDV", "090W33N"), ("1AIWDV", "090W34N"),
                                         ("1AIWDV", "091W35N"), ("1AIWDV", "091W36N"), ("1AIWDV", "091W37N")]}
               }


real_gap_area_filter = {"arizona": {"spatial": [("1AIWDV", "110W36N"), ("1AIWDV", "111W35N"), ("1AIWDV", "111W33N"),
                                                ("1AIWDV", "111W32N"), ("1AIWDV", "112W30N")],
                                    "temporal1": [("1AIWDV", "113W36N"), ("1AIWDV", "113W34N"), ("1AIWDV", "113W33N"),
                                                  ("1AIWDV", "114W31N")],
                                    "temporal2": [("1AIWDV", "109W35N"), ("1AIWDV", "109W34N"), ("1AIWDV", "109W32N"),
                                                  ("1AIWDV","110W31N"), ("1AIWDV", "108W37N")]},
                        "oklahoma": {"spatial": [("1AIWDV", "097W35N"), ("1AIWDV", "097W36N")],
                                     "temporal1": [("1AIWDV", "099W37N"), ("1AIWDV", "099W34N"), ("1AIWDV", "099W36N"),
                                                   ("1AIWDV", "098W33N"), ("1AIWDV", "099W35N"), ("1AIWDV", "098W34N")],
                                     "temporal2": [("1AIWDV", "094W33N"), ("1AIWDV", "095W35N"), ("1AIWDV", "095W36N"),
                                                   ("1AIWDV", "095W34N")]},
                        "iowa": {"spatial": [("1AIWDV", "092W43N"), ("1AIWDV", "092W42N"), ("1AIWDV", "092W41N"),
                                             ("1AIWDV", "092W40N")],
                                 "temporal": [("1AIWDV", "094W43N"), ("1AIWDV", "094W42N"), ("1AIWDV", "094W41N"),
                                              ("1AIWDV", "094W40N"), ("1BIWDV", "092W43N")]},
                        "arkansas": {"spatial": [("1AIWDV", "092W32N"), ("1AIWDV", "092W33N"), ("1AIWDV", "092W34N"),
                                                 ("1AIWDV", "093W35N"), ("1AIWDV", "093W36N"), ("1AIWDV", "093W37N")],
                                     "temporal": [("1AIWDV", "090W32N"), ("1AIWDV", "090W33N"), ("1AIWDV", "090W34N"),
                                                  ("1AIWDV", "091W35N"), ("1AIWDV", "091W36N"), ("1AIWDV", "091W37N")]}
                        }


area_dic = {"iowa": {"lat1": 43.39, "lat2": 40.27, "lon1": -96.44, "lon2": -89.57},
            "oklahoma": {"lat1": 37.06, "lat2": 33.35, "lon1": -103.12, "lon2": -94.17},
            "arkansas": {"lat1": 36.5, "lat2": 32.53, "lon1": -95, "lon2": -89.34,
                         "spatial": ["20180510", "20180521", "20180603", "20180614", "20180708", "20180721", "20180801",
                                     "20180814", "20180825"],
                         "temporal": ["20180410", "20180504", "20180528", "20180621", "20180716", "20180808"]},
            "arizona": {"lat1": 37.5, "lat2": 31.14, "lon1": -115, "lon2": -108.57,
                        "spatial": ["20180507", "20180520", "20180531", "20180613", "20180707", "20180718", "20180731"],
                        "temporal": ["20180405", "20180418", "20180429", "20180512",
                                     "20180523", "20180605", "20180616", "20180710", "20180723"]},
            }


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
                    46: "wisconsin", 47: "wyoming", 48: "maine"}

"""
Croplands: Lands covered with temporary crops followed by harvest and a bare soil period (e.g., single and multiple 
cropping systems). Note that perennial woody crops will be classified as the appropriate forest or shrub land cover 
type.
Cropland/Natural Vegetation Mosaics: Lands with a mosaic of croplands, forest, shrublands, and grasslands in which no 
one component comprises more than 60% of the landscape.
"""
landcover_class_dic = {1: "evergreen_needleleaf_forests",
                       2: "evergreen_broadleaf_forests",
                       3: "deciduous_needleleaf_forests",
                       4: "deciduous_broadleaf_forests",
                       5: "mixed_forests",
                       6: "closed_shrublands",
                       7: "open_shrublands",
                       8: "woody_savannas",
                       9: "savannas",
                       10: "grasslands",
                       11: "permanent_wetlands",
                       12: "croplands",
                       13: "urban_and_built_up_lands",
                       14: "cropland_natural_vegetation_mosaics",
                       15: "permanent_snow_ice",
                       16: "barren",
                       17: "water_bodies",
                       }




# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from .merge_prediction import merge_prediction

__all__ = ["area_dic",
           "merge_prediction"]


area_dic = {
            "iowa": {"lat1": 43.39, "lat2": 40.27, "lon1": -96.44, "lon2": -89.57},
            "oklahoma": {"lat1": 37.06, "lat2": 33.35, "lon1": -103.12, "lon2": -94.17},
            "arizona": {"lat1": 37.5, "lat2": 31.14, "lon1": -115, "lon2": -108.57},
            "arkansas": {"lat1": 36.5, "lat2": 32.53, "lon1": -95, "lon2": -89.34},
            }

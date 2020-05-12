# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from data_preprocessing.utils import load_pkl, save_pkl

import collections
from datetime import datetime


def get_all_doys(pairs):
    all_doys = set()
    for area_pairs in pairs:
        for area in area_pairs:
            for pair in area_pairs[area]:
                for doy in pair:
                    all_doys.add(doy)
    return sorted(list(all_doys))


def get_most_recent_pairs(area_file_lists):
    most_recent_pairs = collections.defaultdict(list)
    for area in ["arizona", "oklahoma"]:
        for doy in area_file_lists[area]["spatial"]:
            temporal1_candidates = [v for v in area_file_lists[area]["temporal1"] if int(v) < int(doy)]
            temporal2_candidates = [v for v in area_file_lists[area]["temporal2"] if int(v) < int(doy)]
            temporal_candi = set()
            if len(temporal1_candidates) > 0 and get_doy_diff(doy, max(temporal1_candidates)) < 36:
                temporal_candi.add(max(temporal1_candidates))
            if len(temporal2_candidates) > 0 and get_doy_diff(doy, max(temporal2_candidates)) < 36:
                temporal_candi.add(max(temporal2_candidates))
            if len(temporal_candi) == 2:
                most_recent_pairs[area].append([doy] + list(temporal_candi))
    for doy in area_file_lists["arkansas"]["spatial"]:
        temporal_candidates = [v for v in area_file_lists["arkansas"]["temporal"] if int(v) < int(doy)]
        if len(temporal_candidates) > 0 and get_doy_diff(doy, max(temporal_candidates)) < 36:
            most_recent_pairs["arkansas"].append((doy, max(temporal_candidates)))
    return most_recent_pairs


def get_doy_diff(doy1, doy2):
    return (datetime(int(doy1[:4]), int(doy1[4:6]), int(doy1[6:]))
            - datetime(int(doy2[:4]), int(doy2[4:6]), int(doy2[6:]))).days


if __name__ == "__main__":
    area_all_files = load_pkl('real_gap_arkansas_file_lists.pkl')

    most_recent_pairs = get_most_recent_pairs(area_all_files)
    all_doys = get_all_doys([most_recent_pairs])
    print(most_recent_pairs)
    print(all_doys)

    save_pkl(most_recent_pairs, "arkansas_real_gap_most_recent_pairs.pkl")
    save_pkl(all_doys, "arkansas_real_gap_all_doys.pkl")


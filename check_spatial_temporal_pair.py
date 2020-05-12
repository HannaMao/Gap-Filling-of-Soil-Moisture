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
                all_doys.add(pair[0])
                all_doys.add(pair[1])
    return sorted(list(all_doys))


def get_most_recent_pairs(area_file_lists):
    most_recent_pairs = collections.defaultdict(list)
    for area in area_file_lists:
        for doy in area_file_lists[area]["spatial"]:
            temporal_candidates = [v for v in area_file_lists[area]["temporal"] if int(v) < int(doy)]
            if len(temporal_candidates) > 0 and get_doy_diff(doy, max(temporal_candidates)) < 36:
                most_recent_pairs[area].append((doy, max(temporal_candidates)))
    return most_recent_pairs


def get_doy_diff(doy1, doy2):
    return (datetime(int(doy1[:4]), int(doy1[4:6]), int(doy1[6:]))
            - datetime(int(doy2[:4]), int(doy2[4:6]), int(doy2[6:]))).days


def get_cohort_1_pairs(area_file_lists):
    cohort1_pairs = collections.defaultdict(list)
    for area in area_file_lists:
        for doy in area_file_lists[area]["spatial"]:
            for temporal_doy in area_file_lists[area]["temporal"]:
                if 0 < get_doy_diff(doy, temporal_doy) < 12:
                    cohort1_pairs[area].append((doy, temporal_doy))
    return cohort1_pairs


def get_cohort_2_pairs(area_file_lists):
    cohort2_pairs = collections.defaultdict(list)
    for area in area_file_lists:
        for doy in area_file_lists[area]["spatial"]:
            for temporal_doy in area_file_lists[area]["temporal"]:
                if 12 < get_doy_diff(doy, temporal_doy) < 24:
                    cohort2_pairs[area].append((doy, temporal_doy))
    return cohort2_pairs


def get_cohort_3_pairs(area_file_lists):
    cohort3_pairs = collections.defaultdict(list)
    for area in area_file_lists:
        for doy in area_file_lists[area]["spatial"]:
            for temporal_doy in area_file_lists[area]["temporal"]:
                if 24 < get_doy_diff(doy, temporal_doy) < 36:
                    cohort3_pairs[area].append((doy, temporal_doy))
    return cohort3_pairs


if __name__ == "__main__":
    area_all_files = load_pkl('arkansas_file_lists.pkl')

    most_recent_pairs = get_most_recent_pairs(area_all_files)
    cohort1_pairs = get_cohort_1_pairs(area_all_files)
    cohort2_pairs = get_cohort_2_pairs(area_all_files)
    cohort3_pairs = get_cohort_3_pairs(area_all_files)
    all_doys = get_all_doys([most_recent_pairs, cohort1_pairs, cohort2_pairs, cohort3_pairs])
    print(most_recent_pairs)
    print(cohort1_pairs)
    print(cohort2_pairs)
    print(cohort3_pairs)
    print(all_doys)

    save_pkl(most_recent_pairs, "arkansas_most_recent_pairs.pkl")
    save_pkl(cohort1_pairs, "arkansas_cohort1_pairs.pkl")
    save_pkl(cohort2_pairs, "arkansas_cohort2_pairs.pkl")
    save_pkl(cohort3_pairs, "arkansas_cohort3_pairs.pkl")
    save_pkl(all_doys, "arkansas_all_doys.pkl")


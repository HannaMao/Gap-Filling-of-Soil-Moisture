# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import numpy as np


def find_index(narr, value):
    new_narr = np.negative(abs(narr - value))
    if max(new_narr) > 0.0001:
        print("ERROR!")
    return np.argmax(new_narr)
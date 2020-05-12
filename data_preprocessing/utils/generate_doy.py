# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from datetime import date, timedelta


def generate_doy(s_doy, e_doy, delimiter):
    s_doy = map(int, [s_doy[:4], s_doy[4:6], s_doy[6:]])
    e_doy = map(int, [e_doy[:4], e_doy[4:6], e_doy[6:]])

    d1 = date(*s_doy)
    d2 = date(*e_doy)
    delta = d2 - d1

    for i in range(delta.days + 1):
        yield str(d1 + timedelta(days=i)).replace("-", delimiter)


def generate_nearest_doys(doy, n, delimiter):
    doy = map(int, [doy[:4], doy[4:6], doy[6:]])
    d1 = date(*doy)

    for i in range((n+1)//2-n, (n+1)//2):
        yield str(d1 + timedelta(days=i)).replace("-", delimiter)


def generate_most_recent_doys(doy, n, delimiter):
    doy = map(int, [doy[:4], doy[4:6], doy[6:]])
    d1 = date(*doy)

    for i in range(-1, -n-1, -1):
        yield str(d1 + timedelta(days=i)).replace("-", delimiter)




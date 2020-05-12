# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

import os
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import ticker
from scipy.spatial import ConvexHull
from matplotlib.colors import ListedColormap, BoundaryNorm
import textwrap as tw

from ..utils import get_out_path
plt.rcParams["font.family"] = "arial"


def get_ax_global():
    # for the EASE-Grid 2.0 which SMAP uses, the projection should be 'cea'
    ax = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180,
                 resolution='c')
    ax.drawparallels(np.arange(-90., 91., 30.), labels=[1, 0, 0, 0], fontsize=10, linewidth=0.1, dashes=[1, 1])
    ax.drawmeridians(np.arange(-180., 181., 60.), labels=[0, 0, 0, 1], fontsize=10, linewidth=0.1, dashes=[1, 1])
    ax.drawcountries(linewidth=0.1)
    ax.drawcoastlines(linewidth=0.1)
    ax.drawstates(linewidth=0.1)

    return ax


def get_ax_usa():
    # for the EASE-Grid 2.0 which SMAP uses, the projection should be 'cea'
    # https: // stackoverflow.com / questions / 42463200 / draw - state - abbreviations - in -matplotlib - basemap
    short_state_names = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        '': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
    }
    ax = Basemap(llcrnrlon=-125,llcrnrlat=24.8,urcrnrlon=-66,urcrnrlat=51,
                 projection='cyl',lat_1=33,lat_2=45,lon_0=-95)
    ax.readshapefile(os.path.join("Data", "US_States", "shapefiles", "st99_d00"), 'states', drawbounds=True)
    printed_names = []
    mi_index = 0
    wi_index = 0
    # for shapedict, state in zip(ax.states_info, ax.states):
    #     draw_state_name = True
    #     short_name = list(short_state_names.keys())[list(short_state_names.values()).index(shapedict['NAME'])]
    #     if short_name in printed_names and short_name not in ['MI', 'WI']:
    #         continue
    #     if short_name == 'MI':
    #         if mi_index != 3:
    #             draw_state_name = False
    #         mi_index += 1
    #     if short_name == 'WI':
    #         if wi_index != 2:
    #             draw_state_name = False
    #         wi_index += 1
    #     hull = ConvexHull(state)
    #     hull_points = np.array(state)[hull.vertices]
    #     # center of convex hull over the polygon points
    #     x, y = hull_points.mean(axis=0)
    #     if draw_state_name:
    #         # You have to align x,y manually to avoid overlapping for little states
    #         plt.text(x + .1, y, short_name, ha="center")
    #     printed_names += [short_name, ]

    # ax.drawparallels(np.arange(25, 65, 20), labels=[1, 0, 0, 0])
    # ax.drawmeridians(np.arange(-120, -40, 20), labels=[0, 0, 0, 1])

    return ax


def get_ax_local(ll_lat, ur_lat, ll_lon, ur_lon):
    # for the EASE-Grid 2.0 which SMAP uses, the projection should be 'cea'
    ax = Basemap(projection='cyl', llcrnrlat=ll_lat, urcrnrlat=ur_lat, llcrnrlon=ll_lon, urcrnrlon=ur_lon)
    ax.drawparallels(np.arange(-90., 91., 1), labels=[1, 0, 0, 0], fontsize=10, linewidth=0.1, dashes=[1, 1])
    ax.drawmeridians(np.arange(-180., 181., 1), labels=[0, 0, 0, 1], fontsize=10, linewidth=0.1, dashes=[1, 1])
    ax.drawcountries()
    ax.drawcoastlines()
    ax.drawstates()

    return ax


def plot_single_variable(in_path,
                         f_names,
                         v_name,
                         out_path,
                         type,
                         unit,
                         fout_name=None,
                         title=None,
                         v_min=None,
                         v_max=None):
    out_path = get_out_path(out_path)

    var_lis = []
    lats, lons = [], []
    for fn in f_names:
        fh = Dataset(os.path.join(in_path, fn + ".nc"), mode="r")
        if len(var_lis) == 0:
            lats = fh.variables['lat'][:]
            lons = fh.variables['lon'][:]
        var_lis.append(fh.variables[v_name][:])
        fh.close()
    average_var = ma.array(var_lis).mean(axis=0)
    vmin = min(average_var.compressed()) if v_min is None else v_min
    vmax = max(average_var.compressed()) if v_max is None else v_max

    lon, lat = np.meshgrid(lons, lats)

    if type == "global":
        ax = get_ax_global()
    elif type == "usa":
        ax = get_ax_usa()
    else:
        ax = get_ax_local(lats[-1], lats[0], lons[0], lons[-1])

    xi, yi = ax(lon, lat)
    cs = ax.pcolor(xi, yi, np.squeeze(average_var), cmap=plt.get_cmap("jet"))
    cs.set_clim(vmin=vmin, vmax=vmax)
    cbar = ax.colorbar(cs, location="bottom", pad="10%")
    cbar.locator = ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()
    cbar.set_label(unit)

    if title is not None:
        title = title
    elif len(f_names) == 1:
        title = f_names[0] + " " + v_name
    else:
        title = f_names[0] + "-" + f_names[-1] + " " + v_name

    if fout_name is not None:
        fout_name = fout_name
    else:
        fout_name = "_".join(title.split(" "))

    plt.title(title, fontsize=12)
    plt.savefig(os.path.join(out_path, fout_name + '.jpg'), dpi=1000)
    plt.close()


def plot_landcover_class(in_path, f_name, out_path, type, title):
    out_path = get_out_path(out_path)

    fh = Dataset(os.path.join(in_path, f_name + ".nc"), mode="r")
    var = fh.variables['landcover_class'][:]
    lats = fh.variables['lat'][:]
    lons = fh.variables['lon'][:]
    fh.close()

    print(np.unique(var, return_counts=True))

    lon, lat = np.meshgrid(lons, lats)

    if type == "global":
        ax = get_ax_global()
    elif type == "usa":
        ax = get_ax_usa()
    else:
        ax = get_ax_local(lats[-1], lats[0], lons[0], lons[-1])

    xi, yi = ax(lon, lat)
    cmap = ListedColormap(["blue",                     # 0  Water   478724
                           "xkcd:forest green",        # 1  Evergreen needleleaf forest   75194
                           "xkcd:grass green",         # 2  Evergreen broadleaf forest    2497
                           "xkcd:light brown",         # 3  Deciduous needleleaf forest   1
                           "xkcd:bright green",        # 4  Deciduous broadleaf forest    48476
                           "xkcd:light forest green",  # 5  Mixed forest                  152871
                           "olive",                    # 6  Closed shrubland              2844
                           "xkcd:puke green",          # 7  Open shrubland                148651
                           "xkcd:mint green",          # 8  Woody savanna                 78988
                           "xkcd:tan",                 # 9  Savanna                       2016
                           "xkcd:leaf",                # 10 Grassland                     293326
                           "khaki",                    # 11 Permanent Wetland             4717
                           "yellow",                   # 12 Croplands                     147569
                           "red",                      # 13 Urban and Built-up            13757
                           "xkcd:navy blue",           # 14 Cropland/Natural Vegetation Mosiac         103096
                           "xkcd:ice",                 # 15 Permanent Snow and Ice        191
                           "xkcd:grey"])        # 16 Barren or Sparsely Vegetated  14057
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5], cmap.N)
    cs = ax.pcolor(xi, yi, np.squeeze(var), cmap=cmap, norm=norm)

    cbar = ax.colorbar(cs, location="bottom", pad="10%", ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15, 16])
    # cbar.ax.set_xticklabels(["Water", "Evergreen needleleaf forest", "Evergreen broadleaf forest",
    #                          "Deciduous needleleaf forest", "Deciduous broadleaf forest", "Mixed forest",
    #                          "Closed shrubland", "Open shrubland", "Woody savanna", "Savanna", "Grassland",
    #                          "Permanent Wetland", "Croplands", "Urban and Built-up",
    #                          "Cropland/Natural Vegetation Mosiac", "Permanent Snow and Ice",
    #                          "Barren or Sparsely Vegetated"])
    cbar.ax.set_xticklabels([*map(str, range(17))])

    plt.title(title, fontsize=12)

    plt.savefig(os.path.join(out_path, 'landcover_class_' + type + '.jpg'), dpi=1000)
    plt.close()


def plot_0_1_variable(in_path, f_name, v_name, out_path, type, on, off):
    out_path = get_out_path(out_path)

    fh = Dataset(os.path.join(in_path, f_name + ".nc"), mode="r")
    var = fh.variables[v_name][:]
    lats = fh.variables['lat'][:]
    lons = fh.variables['lon'][:]
    fh.close()

    lon, lat = np.meshgrid(lons, lats)

    if type == "global":
        ax = get_ax_global()
    elif type == "usa":
        ax = get_ax_usa()
    else:
        ax = get_ax_local(lats[-1], lats[0], lons[0], lons[-1])

    xi, yi = ax(lon, lat)
    cmap = ListedColormap(["blue", "red"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    cs = ax.pcolor(xi, yi, np.squeeze(var), cmap=cmap, norm=norm)

    cbar = ax.colorbar(cs, location="bottom", pad="10%",
                       ticks=[0, 1])
    cbar.ax.set_xticklabels([off, on])

    plt.title(" ".join(v_name.split("_")).title(), fontsize=12)

    plt.savefig(os.path.join(out_path, v_name + '_' + type + '.jpg'), dpi=1000)
    plt.close()







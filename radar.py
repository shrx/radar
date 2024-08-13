#!/usr/bin/python3

import glob
import os
import subprocess
import sys
import pickle
from datetime import datetime

#import cartopy.crs as ccrs
import contextily as ctx
import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import numpy as np
from dateutil import tz
from geopy.geocoders import Nominatim
from pyproj import Transformer

# non-interactive matplotlib
matplotlib.use("Agg")

TRACESTRACK_API_KEY = "YOUR_KEY_HERE"

DPI = 96
PIXEL_SIZE = 400

SRD_PATH = ".local/share/radar"
FRAME_PATH = "/tmp/radar_frames"
CACHE_PATH = ".local/share/nominatim.cache"

geolocator = Nominatim(user_agent="YOUR_USER_AGENT")

cache = dict()
if os.path.exists(CACHE_PATH):
    cache = pickle.load(open(CACHE_PATH, 'rb'))

# ARSO colormap
cmap = ListedColormap(
    [
        "#0061fb",  # 1
        "#008efc",  # 2
        "#00aefb",  # 3
        "#00c7fc",  # 4
        "#30d584",  # 5
        "#56e74c",  # 6
        "#79f62b",  # 7
        "#bdf72b",  # 8
        "#fcf72c",  # 9
        "#ffc223",  # 10
        "#fd8218",  # 11
        "#fd410f",  # 12
        "#d00e07",  # 13
        "#b20f0c",  # 14
        "#c525c8",  # 15
    ]
)

# TODO: read values from .srd header
# https://meteo.arso.gov.si/uploads/meteo/help/sl/SRD3Format.html
# https://meteo.arso.gov.si/uploads/probase/www/observ/radar/si0-zm.srd
# pi     = 3.14159
# deg    = 0.01745
# R      = 6371 km (povprečni zem radij, (a^2*b)^(1/3))
# lon0   = 14.815 * deg (14d48'55"E, GEOSS)
# lat0   = 46.120 * deg (46d07'12"N, GEOSS)
# n      = sin(lat0)
# c      = cos(lat0)*(tan(pi/4+lat0/2))^n/n
# rho0   = c*(tan(pi/4+lat0/2))^(-n)
PROJ_PI = np.pi
PROJ_DEG = PROJ_PI / 180.0
PROJ_R = 6371.0
CENTRAL_LAT = 46.120
CENTRAL_LON = 14.815
PROJ_LON0 = CENTRAL_LON * PROJ_DEG
PROJ_LAT0 = CENTRAL_LAT * PROJ_DEG
PROJ_LAT1 = PROJ_LAT0
PROJ_LAT2 = PROJ_LAT0
PROJ_N = np.sin(PROJ_LAT0)
PROJ_C = (
    np.cos(PROJ_LAT0) * ((np.tan(PROJ_PI / 4.0 + PROJ_LAT0 / 2.0)) ** PROJ_N) / PROJ_N
)
PROJ_RHO0 = PROJ_C * (np.tan(PROJ_PI / 4.0 + PROJ_LAT0 / 2.0)) ** (-PROJ_N)
PROJ_SHIFTX = -4
PROJ_SHIFTY = -6
PROJ_W = 400
PROJ_H = 300
PROJ_X0 = 200.5
PROJ_Y0 = 150.5
STANDARD_PARALLELS = (CENTRAL_LAT, CENTRAL_LAT)

# half map width in km
HALF_W = 10

TEXT_FONT = {
    "family": "serif",
    "color": "k",
    "weight": "normal",
    "size": 10,
    "path_effects": [pe.withStroke(linewidth=2, foreground="white")],
}

if not os.path.exists(FRAME_PATH):
    os.makedirs(FRAME_PATH)


def latLon2xyFloat(lat, lon):
    """
    Converts latitude and longitude to non-rounded ARSO pixel coordinates.

    Args:
        lat (float): Latitude in radians.
        lon (float): Longitude in radians.

    Returns:
        float, float: x and y pixel coordinates
    """
    rho = PROJ_C * (np.tan(PROJ_PI / 4.0 + lat / 2.0)) ** (-PROJ_N)
    x = rho * np.sin(PROJ_N * (lon - PROJ_LON0)) * PROJ_R
    y = (PROJ_RHO0 - rho * np.cos(PROJ_N * (lon - PROJ_LON0))) * PROJ_R
    return PROJ_X0 - PROJ_SHIFTX + x, PROJ_Y0 + PROJ_SHIFTY - y


def xy2latLon(xx, yy):
    """
    Converts ARSO pixel coordinates to latitude and longitude.

    Args:
        xx (float): x pixel coordinate
        yy (float): y pixel coordinate

    Returns:
        float, float: latitude and longitude in degrees
    """
    x = -PROJ_X0 + PROJ_SHIFTX + xx
    y = PROJ_Y0 + PROJ_SHIFTY - yy
    rho = np.sqrt((x / PROJ_R) ** 2.0 + (PROJ_RHO0 - y / PROJ_R) ** 2.0)
    lat = 2.0 * np.arctan((PROJ_C / rho) ** (1.0 / PROJ_N)) - PROJ_PI / 2.0
    lon = PROJ_LON0 + np.arctan((x / PROJ_R) / (PROJ_RHO0 - y / PROJ_R)) / PROJ_N
    return np.degrees(lat), np.degrees(lon)


def calcZoomLevel(lat, mPerPixel=50.0):
    """
    Calculates appropriate zoom level of the OSM base tile map.

    Args:
        lat (float): latitude in degrees
        mPerPixel (float): desired map tile resolution in meters per pixel

    Returns:
        int: OSM map tile zoom level
    """
    # https://wiki.openstreetmap.org/wiki/Zoom_levels
    return round(
        np.log2(2 * np.pi * 6378137.000 * np.cos(np.radians(lat)) / mPerPixel) - 8.0
    )


def timestampText(s):
    """
    Parses SRD time field and converts it to a timestamp to be shown on the map.

    Args:
        s (string): SRD time field

    Returns:
        string: timestamp text
    """
    # 2024 08 01 17 30
    return (
        datetime.strptime(s, "%Y %m %d %H %M")
        .replace(tzinfo=tz.tzutc())
        .astimezone(tz.tzlocal())
        .strftime("%-d. %-m. %-H:%M")
    )


def name2latlon(name):
    """
    Converts a location name to its corresponding geo coordinates.

    Args:
        name (string): location name

    Returns:
        float, float: latitude and longitude in degrees
    """
    location = geolocator.geocode(name)
    if not location:
        print("location not found")
        sys.exit(1)
    return float(location.raw["lat"]), float(location.raw["lon"])


def getCoords(inp):
    """
    Converts a location name to its corresponding geo coordinates.

    Args:
        inp (string): location name input

    Returns:
        float, float: latitude and longitude in degrees
    """
    loc = " ".join(inp)
    if loc not in cache:
        cache[loc] = name2latlon(loc)
        print("cache updated")
        pickle.dump(cache, open(CACHE_PATH, 'wb'))
    return cache[loc]


def generateAnimation(lat, lon):
    """
    Generates a radar precipitation animation for the given coordinates.

    Args:
        lat (float): latitude in degrees
        lon (float): longitude in degrees
    """
    # compute extents
    x, y = latLon2xyFloat(np.radians(lat), np.radians(lon))
    xInt, yInt = round(x), round(y)
    latMin, lonMin = xy2latLon(x - HALF_W, y + HALF_W)
    latMax, lonMax = xy2latLon(x + HALF_W, y - HALF_W)
    extentRadar = [xInt - HALF_W - 1, xInt + HALF_W + 2, yInt + HALF_W + 2, yInt - HALF_W - 1]

    # init empty array
    dataRadar = np.zeros(shape=(2 * HALF_W + 3, 2 * HALF_W + 3)) * np.nan

    zoomLevel = calcZoomLevel(lat)

    # parse SRD data
    srdFiles = glob.glob(os.path.join(SRD_PATH, "*.srd"))
    srdFiles = [f for f in srdFiles if os.path.isfile(f)]
    srdFiles.sort(key=os.path.getmtime)
    # get last 90 minutes
    srdFiles = srdFiles[-(90 // 5) :]
    srds = []
    for i, srd in enumerate(srdFiles):
        print(f"Parsing SRD {i+1}...", end="\r")
        with open(srd, "r", encoding="ascii") as srdFile:
            srdText = srdFile.read().splitlines()

        srdTime = srdText[4].split("      ", 1)[1]
        dataStart = srdText.index("DATA") + 1
        dataText = srdText[dataStart:]
        dataRaw = np.array([list(map(ord, line)) for line in dataText], dtype=float)
        dataZm = 3.0 * (dataRaw - 64) + 12.0
        # exclude NODATA
        dataZm[dataRaw == 126.0] = np.nan
        # exclude no precipitation
        dataZm[dataRaw == 64.0] = np.nan

        dataRadar = dataZm[
            yInt - HALF_W - 1 : yInt + HALF_W + 2,
            xInt - HALF_W - 1 : xInt + HALF_W + 2,
        ]
        srds.append([srdTime, dataRadar])
    print("Finished parsing SRD files.")
    if np.all(np.isnan([srd[1] for srd in srds])):
        print("no rain")
        sys.exit()

    # populate plot elements
    # base figure
    fig = plt.figure(figsize=(PIXEL_SIZE / DPI, PIXEL_SIZE / DPI), dpi=DPI)
    ax = plt.axes(frameon=False)
    ax.set_axis_off()
    ax.set_xlim(x - HALF_W, x + HALF_W)
    ax.set_ylim(y + HALF_W, y - HALF_W)
    # OSM map tile   
    #basemap, basemap_extent = ctx.bounds2img(lonMin, latMin, lonMax, latMax, zoom=zoomLevel, source=ctx.providers.OpenStreetMap.Mapnik, ll=True)
    basemap, basemap_extent = ctx.bounds2img(lonMin, latMin, lonMax, latMax, zoom=zoomLevel, source="https://tile.tracestrack.com/_/{z}/{x}/{y}.png?key=" + TRACESTRACK_API_KEY, ll=True)
    # no need to bother with reprojection to LCC on such a small scale
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326")
    min_lat_t, min_lon_t = transformer.transform(basemap_extent[0],basemap_extent[2])
    max_lat_t, max_lon_t = transformer.transform(basemap_extent[1],basemap_extent[3])
    min_x_t, max_y_t = latLon2xyFloat(np.radians(min_lat_t), np.radians(min_lon_t))
    max_x_t, min_y_t = latLon2xyFloat(np.radians(max_lat_t), np.radians(max_lon_t))
    ax.imshow(basemap, zorder=1, extent=[min_x_t, max_x_t, min_y_t, max_y_t], origin="lower")
    
    # radar data layer
    radarLayer = ax.imshow(
        dataRadar,
        cmap=cmap,
        vmin=15.0,
        vmax=57.0,
        interpolation="none",
        alpha=0.5,
        zorder=2,
        extent=extentRadar,
    )
    # concentric circles
    for rKm in [2, 5, 10]:
        circle = Circle((x, y), radius=rKm, facecolor="none", edgecolor="k", zorder=3)
        ax.add_patch(circle)
    
    # time label
    timeText = ax.text(
        0.99, 0.99, "", fontdict=TEXT_FONT, ha="right", va="top", transform=ax.transAxes, zorder=4
    )
    
    ctx.add_attribution(ax, "© OpenStreetMap Contributors")

    ax.set_frame_on(False)
    plt.tight_layout(pad=0)
    
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)

    # draw frames
    for i, srd in enumerate(srds):
        srdTime = srd[0]
        print(f"Rendering frame {i+1} ({srdTime})...", end="\r")
        
        fig.canvas.restore_region(background)
        
        dataRadar = srd[1]
        radarLayer.set_data(dataRadar)
        radarLayer.set_extent(extentRadar)
        timeText.set_text(f"{timestampText(srdTime)}")
        
        ax.draw_artist(radarLayer)
        ax.draw_artist(timeText)
        fig.canvas.blit(ax.bbox)
        
        plt.savefig(f"{FRAME_PATH}/frame_{i:03}.png", bbox_inches=0, pad_inches=0)

    # render animation
    # webm output:
    # ffmpeg -y -framerate 30 -pattern_type glob -i f"{FRAME_PATH}/frame_*.png" -c:v libvpx -pix_fmt yuv420p", f"{FRAME_PATH}/radar.webm"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-pattern_type",
            "glob",
            "-i",
            f"{FRAME_PATH}/frame_*.png",
            "-vf",
            "palettegen",
            f"{FRAME_PATH}/palette.png",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # https://stackoverflow.com/questions/46952350/ffmpeg-concat-demuxer-with-duration-filter-issue
    # https://trac.ffmpeg.org/ticket/6128
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "4",
            "-pattern_type",
            "glob",
            "-i",
            f"{FRAME_PATH}/frame_*.png",
            "-i",
            f"{FRAME_PATH}/palette.png",
            "-filter_complex",
            f"[0:v]split=2[main][dup];[dup]trim=start_frame={len(srds)-1}:end_frame={len(srds)},setpts=PTS-STARTPTS[dupfirst];[dupfirst][main]concat=n=2:v=1:a=0[allframes];[allframes][1:v]paletteuse",
            "-final_delay",
            f"{round((2.5 - 0.25) * 100)}",
            f"{FRAME_PATH}/radar.gif",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for frame in glob.glob(f"{FRAME_PATH}/frame_*.png"):
        os.remove(frame)

    print("Finished rendering frames.")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        latitude, longitude = getCoords(sys.argv[1:])
        max_lat, min_lon = xy2latLon(0, 0)
        min_lat, max_lon = xy2latLon(PROJ_W, PROJ_H)
        if latitude < min_lat or latitude > max_lat or longitude < min_lon or longitude > max_lon:
            print("location is out of radar range!")
            sys.exit(1)
        generateAnimation(latitude, longitude)


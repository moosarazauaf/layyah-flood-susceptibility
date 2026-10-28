"""
Shared Earth Engine definitions for the Layyah flood-susceptibility study.

Design: a model is trained on the Indus floodplain of six Punjab districts
and tested on Layyah, which it never sees during training. Flood labels are
optical only (Landsat, plus Sentinel-2 from 2022), 2000-2025.

Study domain: land within CORRIDOR_M of the Indus main stem in each
district. The main stem is MERIT Hydro upstream area >= 300,000 km2: the
Indus carries 316,000 km2 at Kalabagh and 401,000 km2 at Layyah, the Chenab
154,000 km2 east of Muzaffargarh and the Panjnad 279,000 km2, so the cut
keeps the Indus and excludes the Chenab-Panjnad system that bounds
Muzaffargarh to the east. Hill torrents of the Suleman range lie mostly
beyond the corridor.
"""
import json
import os

import ee

TRAIN_DISTRICTS = ["Mianwali District", "Bhakkar District", "Muzaffargarh District",
                   "Dera Ghazi Khan District", "Rajanpur District",
                   "Rahim Yar Khan District"]
TEST_DISTRICT = "Layyah District"

INDUS_MIN_UPA_KM2 = 300_000
CORRIDOR_M = 15_000
DIST_SCALE_M = 90          # MERIT native; distance transforms are pinned to it
DIST_NEIGHBOURHOOD_PX = 512   # search radius: 512 x 90 m = 46 km

TRAIN_YEARS = list(range(2000, 2020))      # flood inventory used for training
# Temporal test: 2020-2021, the last JRC years, so labels never change
# method mid-test. 2022-2025 use the continuation method and form a
# supplementary test: it agrees with JRC at IoU 0.48-0.57 on 2019-2021
# (outputs/label_sensor_test.json), too low to mix the two silently.
TEMPORAL_YEARS = [2020, 2021]
RECENT_YEARS = [2022, 2023, 2024, 2025]
JRC_LAST_YEAR = 2021                       # JRC monthly history ends here
MONSOON = (7, 10)                          # Jul-Oct
PRE_MONSOON = (3, 5)                       # Mar-May
MIN_OBS_YEARS = 10                         # of 20 training years
# sensors for the 2022+ labels; set after the JRC agreement test in
# scripts/01b_label_sensor_test.py
# Landsat 8/9 only: mean IoU with JRC 0.52, against 0.46 with Sentinel-2
# added (outputs/label_sensor_test.json)
OWN_SENSORS = ("landsat",)
PERMANENT_OCCURRENCE = 75


def init_ee():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(root, ".env"))
    except ImportError:
        pass
    key = os.environ.get("EE_SERVICE_ACCOUNT_KEY")
    project = os.environ.get("EE_PROJECT")
    if key and os.path.isfile(key):
        email = json.load(open(key, encoding="utf-8"))["client_email"]
        ee.Initialize(ee.ServiceAccountCredentials(email, key), project=project)
    else:
        ee.Initialize(project=project)


def district(name):
    return (ee.FeatureCollection("FAO/GAUL/2015/level2")
            .filter(ee.Filter.eq("ADM0_NAME", "Pakistan"))
            .filter(ee.Filter.eq("ADM2_NAME", name)).geometry())


def all_districts():
    return ee.FeatureCollection([ee.Feature(district(n)) for n in
                                 TRAIN_DISTRICTS + [TEST_DISTRICT]]).geometry()


def indus():
    return (ee.Image("MERIT/Hydro/v1_0_1").select("upa")
            .gte(INDUS_MIN_UPA_KM2).selfMask().rename("indus"))


def distance_to(mask_img):
    """Metres to the nearest pixel of mask_img, pinned to DIST_SCALE_M.

    fastDistanceTransform works in whatever grid it inherits and can return
    a near-constant surface without raising; check_distance() guards that.
    It also returns a junk value (thousands of km) beyond its search radius
    instead of failing, so those pixels are masked."""
    d = (mask_img.unmask(0).fastDistanceTransform(DIST_NEIGHBOURHOOD_PX).sqrt()
         .multiply(DIST_SCALE_M))
    return (d.updateMask(d.lte(DIST_NEIGHBOURHOOD_PX * DIST_SCALE_M))
            .reproject(crs="EPSG:4326", scale=DIST_SCALE_M))


def check_distance(dist, region):
    s = dist.reduceRegion(ee.Reducer.percentile([5, 50, 95]), region,
                          DIST_SCALE_M, maxPixels=1e10, bestEffort=True).getInfo()
    vals = list(s.values())
    if None in vals or max(vals) - min(vals) < 2000:
        raise RuntimeError(f"distance surface degenerate: {s}")
    return s


def corridor_mask():
    # beyond the search radius the distance is masked; that land is far
    # outside the corridor, so it becomes 0, not missing
    return distance_to(indus()).lte(CORRIDOR_M).unmask(0).rename("corridor")


def permanent_water():
    return (ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
            .unmask(0).gte(PERMANENT_OCCURRENCE))


# ---------------------------------------------------------------------------
# Flood labels
# ---------------------------------------------------------------------------

def _flood_rule(wet_water, wet_valid, dry_water, dry_valid):
    """Flooded: water in a monsoon month at a pixel dry in every observed
    pre-monsoon month of the same year. The pre-monsoon test stops channel
    water that existed before the monsoon being counted as flooding."""
    observed = wet_valid.And(dry_valid)
    return wet_water.And(dry_water.Not()).And(observed), observed


def jrc_flood_year(year):
    """Landsat flood year, 2000-2021, from the JRC monthly water history
    (water band: 0 no data, 1 not water, 2 water)."""
    mh = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory").filter(ee.Filter.eq("year", year))
    sel = lambda a, b: mh.filter(ee.Filter.rangeContains("month", a, b))
    w = lambda c: c.map(lambda i: i.select("water").eq(2)).max().unmask(0)
    v = lambda c: c.map(lambda i: i.select("water").gt(0)).max().unmask(0)
    wet, dry = sel(*MONSOON), sel(*PRE_MONSOON)
    return _flood_rule(w(wet), v(wet), w(dry), v(dry))


def _landsat_mndwi(img):
    qa = img.select("QA_PIXEL")
    clear = (qa.bitwiseAnd(1 << 1).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
             .And(qa.bitwiseAnd(1 << 4).eq(0)))          # dilated cloud, cloud, shadow
    sr = img.select(["SR_B3", "SR_B6"]).multiply(0.0000275).add(-0.2)
    return sr.normalizedDifference(["SR_B3", "SR_B6"]).rename("mndwi").updateMask(clear)


def _s2_mndwi(img):
    clear = img.select("SCL").remap([2, 4, 5, 6, 7], [1, 1, 1, 1, 1], 0)
    return img.normalizedDifference(["B3", "B11"]).rename("mndwi").updateMask(clear)


def optical_month(year, month, region, sensors=OWN_SENSORS):
    """Monthly water from Landsat 8/9 and/or Sentinel-2: (water, valid).
    Water is the median MNDWI of all clear observations in the month > 0,
    mirroring JRC's monthly classification."""
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    ls = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
          .filterBounds(region).filterDate(start, end).map(_landsat_mndwi))
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region).filterDate(start, end).map(_s2_mndwi))
    coll = {"landsat": ls, "s2": s2}
    parts = [coll[k] for k in sensors]
    m = (parts[0].merge(parts[1]) if len(parts) == 2 else parts[0]).median()
    return m.gt(0).unmask(0), m.mask().unmask(0)


def own_flood_year(year, region, sensors=OWN_SENSORS):
    """Same rule as jrc_flood_year, from Landsat 8/9 + Sentinel-2 monthly
    composites; used for 2022 onward, where the JRC record ends."""
    def months(a, b):
        ws, vs = zip(*(optical_month(year, m, region, sensors) for m in range(a, b + 1)))
        return (ee.ImageCollection(list(ws)).max(), ee.ImageCollection(list(vs)).max())
    ww, wv = months(*MONSOON)
    dw, dv = months(*PRE_MONSOON)
    return _flood_rule(ww, wv, dw, dv)


def flood_year(year, region):
    return jrc_flood_year(year) if year <= JRC_LAST_YEAR else own_flood_year(year, region)


# ---------------------------------------------------------------------------
# Predictors: none uses imagery from a flood season
# ---------------------------------------------------------------------------

PREDICTORS = ["hand", "elev", "slope", "dist_indus", "dist_drain", "twi",
              "clay", "rain", "ndvi_dry", "ndbi_dry"]


def predictors():
    merit = ee.Image("MERIT/Hydro/v1_0_1")
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic() \
        .setDefaultProjection("EPSG:4326", None, 30)
    slope = ee.Terrain.slope(dem)
    upa = merit.select("upa")                          # km2
    # topographic wetness index: ln(specific catchment area / tan slope)
    sca = upa.multiply(1e6).divide(90)
    twi = sca.divide(slope.multiply(3.14159265 / 180).tan().max(0.001)).log()
    rain = (ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD")
            .filterDate("2000-01-01", "2020-01-01")
            .filter(ee.Filter.calendarRange(*MONSOON, "month"))
            .sum().divide(len(TRAIN_YEARS)))          # mean monsoon total, mm
    # dry-season (Nov-Feb) land-surface indices, Landsat 8, within the
    # training period; never from flood-season imagery, which would leak
    # the answer (flood-season water indices ARE the flood)
    def l8_idx(img):
        qa = img.select("QA_PIXEL")
        clear = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        sr = img.select(["SR_B4", "SR_B5", "SR_B6"]).multiply(0.0000275).add(-0.2)
        return ee.Image.cat([sr.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi_dry"),
                             sr.normalizedDifference(["SR_B6", "SR_B5"]).rename("ndbi_dry")]
                            ).updateMask(clear)
    dry = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
           .filterDate("2013-04-01", "2020-01-01")
           .filter(ee.Filter.Or(ee.Filter.calendarRange(11, 12, "month"),
                                ee.Filter.calendarRange(1, 2, "month")))
           .map(l8_idx).median())
    clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select("b0")
    return ee.Image.cat([
        merit.select("hnd").rename("hand"),
        dem.rename("elev"),
        slope.rename("slope"),
        distance_to(indus()).rename("dist_indus"),
        distance_to(upa.gte(100)).rename("dist_drain"),
        twi.rename("twi"),
        clay.rename("clay"),
        rain.rename("rain"),
        dry,
    ]).select(PREDICTORS).toFloat()

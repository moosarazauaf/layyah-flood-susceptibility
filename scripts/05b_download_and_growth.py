"""
Two follow-ups to 05_map_layyah.py.

1. Download the susceptibility score at 30 m in latitude strips. A single
   30 m request over Layyah returns HTTP 400 (the predictor stack is too
   heavy for one synchronous request); strips stay within the limit.
   make_maps_qgis.py mosaics them.

2. Population growth context. People on class 4-5 land roughly tripled
   2000-2020 (outputs/layyah_map_stats.json). Before that can be called
   movement into flood-prone land it is compared with growth across the
   corridor and across each whole tehsil; if everything tripled, the high-
   class figure only reflects general growth or GHSL's redistribution.
"""
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.request

import ee
import joblib

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import fs_common as F

spec = importlib.util.spec_from_file_location(
    "m5", os.path.join(_ROOT, "scripts", "05_map_layyah.py"))
m5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m5)

OUT = os.path.join(_ROOT, "outputs")
STRIPS = 6
MAX_DEPTH = 2


def fetch(url, path, tries=4):
    for a in range(1, tries + 1):
        try:
            urllib.request.urlretrieve(url, path); return True
        except (urllib.error.HTTPError, OSError) as e:
            print(f"  retry {a}: {str(e)[:120]}"); time.sleep(15 * a)
    return False


def main():
    F.init_ee()
    layyah = F.district(F.TEST_DISTRICT)
    keep = F.corridor_mask().And(F.permanent_water().Not())
    score = m5.ee_logistic(joblib.load(os.path.join(OUT, "model_logistic.joblib")))
    score = score.updateMask(keep).clip(layyah)

    # 1. strips; a strip that fails is split into quarters and retried,
    # down to MAX_DEPTH, so only the heavy corridor pieces get small
    d = os.path.join(OUT, "score_strips"); os.makedirs(d, exist_ok=True)
    b = layyah.bounds(10).coordinates().get(0).getInfo()
    x0, x1 = min(p[0] for p in b), max(p[0] for p in b)
    y0, y1 = min(p[1] for p in b), max(p[1] for p in b)
    img = score.multiply(100).round().unmask(255).toUint8()

    def get(tag, lo, hi, depth):
        path = os.path.join(d, f"strip_{tag}.tif")
        if os.path.isfile(path):
            print(f"strip {tag}: already downloaded"); return
        box = ee.Geometry.Rectangle([x0, lo, x1, hi], None, False)
        url = img.getDownloadURL({"region": box, "scale": 30, "crs": "EPSG:4326",
                                  "format": "GEO_TIFF"})
        if fetch(url, path, tries=2):
            print(f"strip {tag}: ok"); return
        if depth >= MAX_DEPTH:
            raise SystemExit(f"strip {tag} still fails at depth {depth}")
        print(f"strip {tag}: splitting into 4")
        for j in range(4):
            get(f"{tag}-{j}", lo + (hi - lo) * j / 4, lo + (hi - lo) * (j + 1) / 4, depth + 1)

    for i in range(STRIPS):
        lo, hi = y0 + (y1 - y0) * i / STRIPS, y0 + (y1 - y0) * (i + 1) / STRIPS
        get(str(i), lo, hi, 0)

    # 2. growth context: whole tehsil, corridor, classes 4-5
    if os.path.isfile(os.path.join(OUT, "population_growth.json")):
        print("population_growth.json: already done"); return
    stats = json.load(open(os.path.join(OUT, "layyah_map_stats.json"), encoding="utf-8"))
    teh = json.load(open(os.path.join(_ROOT, "data", "layyah_tehsils.geojson"), encoding="utf-8"))
    growth = {}
    for f in teh["features"]:
        name = f["properties"]["adm3_name"]
        g = ee.Geometry(f["geometry"]).intersection(layyah, 10)
        tot = {}
        for yr in (2000, 2010, 2020):
            v = (ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{yr}").select("population_count")
                 .reduceRegion(ee.Reducer.sum(), g, 100, maxPixels=1e11,
                               bestEffort=True, tileScale=4).get("population_count").getInfo())
            tot[yr] = round(v or 0)
        e = stats["exposure"].get(name, {})
        corridor = {yr: round(sum(e.get(f"pop_{yr}", {}).values())) for yr in (2000, 2010, 2020)}
        high = {yr: round(sum(e.get(f"pop_{yr}", {}).get(str(c), 0) for c in (4, 5)))
                for yr in (2000, 2010, 2020)}
        ratio = lambda r: round(r[2020] / r[2000], 2) if r[2000] else None
        growth[name] = {"tehsil": tot, "corridor": corridor, "classes_4_5": high,
                        "ratio_2020_2000": {"tehsil": ratio(tot), "corridor": ratio(corridor),
                                            "classes_4_5": ratio(high)}}
        print(f"{name:<16} 2020/2000 population ratio: tehsil {ratio(tot)} | "
              f"corridor {ratio(corridor)} | classes 4-5 {ratio(high)}")
    json.dump(growth, open(os.path.join(OUT, "population_growth.json"), "w", encoding="utf-8"),
              indent=2)
    print("wrote outputs/population_growth.json")


if __name__ == "__main__":
    main()

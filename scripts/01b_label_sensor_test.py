"""
Which sensors should the 2022+ flood labels use?

The JRC record (Landsat only, expert-system classifier) ends in 2021. For
2019-2021, flood-year maps from the continuation method are compared with
JRC over Layyah's corridor, with Landsat 8/9 only and with Landsat 8/9 plus
Sentinel-2. The 5-day Sentinel-2 revisit sees short-lived water that a
Landsat-only record misses, which may explain the moderate agreement found
in 01_check_inputs.py. The variant closer to JRC is used for 2022-2025.

Writes outputs/label_sensor_test.json.
"""
import json
import os
import sys

import ee

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import fs_common as F

SCALE = 30


def km2(mask, geom):
    v = ee.Image.pixelArea().updateMask(mask.selfMask()).reduceRegion(
        ee.Reducer.sum(), geom, SCALE, maxPixels=1e11, bestEffort=True,
        tileScale=4).get("area")
    return ee.Number(ee.Algorithms.If(v, v, 0)).divide(1e6).getInfo()


def main():
    F.init_ee()
    region = F.district(F.TEST_DISTRICT)
    keep = F.corridor_mask().And(F.permanent_water().Not())
    rows = []
    for y in (2019, 2020, 2021):
        jf, jo = F.jrc_flood_year(y)
        for sensors in (("landsat",), ("landsat", "s2")):
            of, oo = F.own_flood_year(y, region, sensors)
            obs = jo.And(oo).And(keep)
            j, o = km2(jf.And(obs), region), km2(of.And(obs), region)
            both = km2(jf.And(of).And(obs), region)
            iou = both / (j + o - both) if (j + o - both) else None
            rows.append({"year": y, "sensors": "+".join(sensors), "jrc_km2": round(j, 1),
                         "own_km2": round(o, 1), "agree_km2": round(both, 1),
                         "iou": round(iou, 3) if iou is not None else None})
            print(f"{y} {'+'.join(sensors):<11} JRC {j:6.1f} own {o:6.1f} "
                  f"agree {both:6.1f} IoU {rows[-1]['iou']}")
            json.dump(rows, open(os.path.join(_ROOT, "outputs", "label_sensor_test.json"),
                                 "w", encoding="utf-8"), indent=2)
    print("wrote outputs/label_sensor_test.json")


if __name__ == "__main__":
    main()

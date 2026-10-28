"""
Check the foundations before any heavy computation.

1. All seven districts resolve in GAUL 2015, with plausible areas.
2. The 15 km Indus corridor and the distance surface it rests on are sane
   (fastDistanceTransform can return a near-constant surface silently).
3. The labelling method used from 2022 onward (Landsat 8/9 + Sentinel-2
   monthly MNDWI) agrees with the JRC record it continues. For 2019-2021,
   where both exist, flood-year maps from each are compared over Layyah's
   corridor. If they disagree, the 2022-2025 labels and the temporal test
   built on them cannot be trusted.

Writes outputs/check_inputs.json.
"""
import json
import os
import sys

import ee

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import fs_common as F

SCALE = 30


def km2(mask, geom, scale=SCALE):
    v = ee.Image.pixelArea().updateMask(mask.selfMask()).reduceRegion(
        ee.Reducer.sum(), geom, scale, maxPixels=1e11, bestEffort=True,
        tileScale=4).get("area")
    return ee.Number(ee.Algorithms.If(v, v, 0)).divide(1e6).getInfo()


def main():
    F.init_ee()
    out = {"districts": {}}
    corr = F.corridor_mask()
    for name in F.TRAIN_DISTRICTS + [F.TEST_DISTRICT]:
        g = F.district(name)
        total = g.area(10).divide(1e6).getInfo()
        c = km2(corr, g, F.DIST_SCALE_M)
        out["districts"][name] = {"area_km2": round(total, 1), "corridor_km2": round(c, 1)}
        print(f"{name:<26} area {total:8.0f} km2 | corridor {c:7.0f} km2")

    layyah = F.district(F.TEST_DISTRICT)
    out["distance_check_layyah"] = F.check_distance(F.distance_to(F.indus()), layyah)
    print("distance to Indus, Layyah p5/p50/p95 (m):", out["distance_check_layyah"])

    region = layyah
    lc = corr.unmask(0)
    perm = F.permanent_water()
    print("\nJRC vs own method, flood-year maps over Layyah corridor:")
    rows = []
    for y in (2019, 2020, 2021):
        jf, jo = F.jrc_flood_year(y)
        of, oo = F.own_flood_year(y, region)
        both_obs = jo.And(oo).And(lc).And(perm.Not())
        a = {"year": y,
             "both_observed_km2": km2(both_obs, region),
             "jrc_flood_km2": km2(jf.And(both_obs), region),
             "own_flood_km2": km2(of.And(both_obs), region),
             "agree_flood_km2": km2(jf.And(of).And(both_obs), region)}
        union = a["jrc_flood_km2"] + a["own_flood_km2"] - a["agree_flood_km2"]
        a["iou"] = round(a["agree_flood_km2"] / union, 3) if union else None
        a = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in a.items()}
        rows.append(a)
        print(f"  {y}: observed {a['both_observed_km2']:7.1f} | JRC flood "
              f"{a['jrc_flood_km2']:6.1f} | own flood {a['own_flood_km2']:6.1f} | "
              f"agree {a['agree_flood_km2']:6.1f} | IoU {a['iou']}")
        json.dump({**out, "label_consistency": rows},
                  open(os.path.join(_ROOT, "outputs", "check_inputs.json"), "w",
                       encoding="utf-8"), indent=2)
    print("wrote outputs/check_inputs.json")


if __name__ == "__main__":
    main()

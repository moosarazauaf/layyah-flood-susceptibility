"""
Earth Engine analyses requested in the peer review (paper/review/review_2026-10-26.md).
Each part writes its own JSON under outputs/revision/ and is skipped if present,
so an interrupted run resumes.

  points.json     per Layyah test point: flooded in 2010, years flooded excluding
                  2010, ESA WorldCover class                         (R1-5, R2-1)
  per_year.json   flooded area per year 2000-2021 in the Layyah corridor, and
                  area flooded only in 2010                          (R2-1)
  bands.json      corridor area and flooded area by distance-to-Indus band;
                  capture of flooded land in the nearest 40% of the corridor by
                  distance alone                                     (R1-3, R3-2)
  buildings.json  Google Open Buildings Temporal, 2016 vs 2023, by susceptibility
                  class group and tehsil: independent check on the GHSL
                  population-growth claim                           (DA-1)
"""
import importlib.util
import json
import os
import sys

import ee

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import fs_common as F

OUT = os.path.join(_ROOT, "outputs", "revision")
SCALE = 30
os.makedirs(OUT, exist_ok=True)


def cached(name, fn):
    path = os.path.join(OUT, name)
    if os.path.isfile(path):
        print(f"{name}: cached"); return json.load(open(path, encoding="utf-8"))
    res = fn()
    json.dump(res, open(path, "w", encoding="utf-8"), indent=2)
    print(f"{name}: written"); return res


def km2(mask, geom, scale=SCALE):
    v = ee.Image.pixelArea().updateMask(mask.selfMask()).reduceRegion(
        ee.Reducer.sum(), geom, scale, maxPixels=1e11, bestEffort=True,
        tileScale=8).get("area")
    return ee.Number(ee.Algorithms.If(v, v, 0)).divide(1e6).getInfo()


def flood_images():
    f = {y: F.jrc_flood_year(y)[0] for y in F.TRAIN_YEARS + F.TEMPORAL_YEARS}
    return f


def part_points(layyah, flood):
    others = ee.ImageCollection([flood[y] for y in F.TRAIN_YEARS if y != 2010]).sum()
    img = ee.Image.cat([flood[2010].rename("f2010"), others.rename("n_excl2010"),
                        ee.Image("ESA/WorldCover/v200/2021").select("Map").rename("wc")])
    rows = []
    for tag in ("spatial", "temporal"):
        pts = json.load(open(os.path.join(_ROOT, "outputs", "samples",
                                          f"test_{tag}_Layyah.json"), encoding="utf-8"))
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(p["lon"], p["lat"]), {"i": i})
                                   for i, p in enumerate(pts)])
        got = {f["properties"]["i"]: f["properties"] for f in
               img.sampleRegions(fc, scale=SCALE, tileScale=4).getInfo()["features"]}
        for i, p in enumerate(pts):
            g = got.get(i, {})
            rows.append({"set": tag, "i": i, "lon": p["lon"], "lat": p["lat"], "label": p["label"],
                         "f2010": g.get("f2010"), "n_excl2010": g.get("n_excl2010"),
                         "wc": g.get("wc")})
    return rows


def part_per_year(layyah, flood):
    keep = F.corridor_mask().And(F.permanent_water().Not())
    out = {"corridor_km2": round(km2(keep, layyah), 1), "years": {}}
    for y, img in sorted(flood.items()):
        out["years"][y] = round(km2(img.And(keep), layyah), 1)
    others = ee.ImageCollection([flood[y] for y in F.TRAIN_YEARS if y != 2010]).max()
    out["only_2010_km2"] = round(km2(flood[2010].And(others.Not()).And(keep), layyah), 1)
    out["any_2000_2019_km2"] = round(km2(ee.ImageCollection(
        [flood[y] for y in F.TRAIN_YEARS]).max().And(keep), layyah), 1)
    out["any_excl_2010_km2"] = round(km2(others.And(keep), layyah), 1)
    return out


def part_bands(layyah, flood):
    keep = F.corridor_mask().And(F.permanent_water().Not())
    dist = F.distance_to(F.indus())
    f_any = ee.ImageCollection([flood[y] for y in F.TRAIN_YEARS]).max()
    f_new = ee.ImageCollection([flood[y] for y in F.TEMPORAL_YEARS]).max().And(f_any.Not())
    edges = [0, 1000, 2000, 3000, 5000, 8000, 15000]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = keep.And(dist.gte(lo)).And(dist.lt(hi))
        rows.append({"from_km": lo / 1000, "to_km": hi / 1000,
                     "corridor_km2": round(km2(band, layyah), 1),
                     "flooded_2000_2019_km2": round(km2(band.And(f_any), layyah), 1),
                     "first_flooded_2020_2021_km2": round(km2(band.And(f_new), layyah), 2)})
    # nearest 40% of the corridor by distance
    p = dist.updateMask(keep).reduceRegion(ee.Reducer.percentile([40]), layyah, SCALE,
                                           maxPixels=1e11, bestEffort=True, tileScale=8).getInfo()
    cut = list(p.values())[0]
    near = keep.And(dist.lte(cut))
    out = {"bands": rows, "p40_distance_m": round(cut),
           "nearest40_corridor_km2": round(km2(near, layyah), 1),
           "nearest40_flooded_2000_2019_km2": round(km2(near.And(f_any), layyah), 1),
           "nearest40_first_flooded_2020_2021_km2": round(km2(near.And(f_new), layyah), 2),
           "all_flooded_2000_2019_km2": round(km2(keep.And(f_any), layyah), 1),
           "all_first_flooded_2020_2021_km2": round(km2(keep.And(f_new), layyah), 2)}
    return out


def part_buildings(layyah):
    spec = importlib.util.spec_from_file_location(
        "m5", os.path.join(_ROOT, "scripts", "05_map_layyah.py"))
    m5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m5)
    import joblib
    model = joblib.load(os.path.join(_ROOT, "outputs", "model_logistic.joblib"))
    keep = F.corridor_mask().And(F.permanent_water().Not())
    score = m5.ee_logistic(model).updateMask(keep).clip(layyah)
    cuts = json.load(open(os.path.join(_ROOT, "outputs", "layyah_map_stats.json")))["quintile_cuts"]
    c45 = score.gt(cuts[2])                 # above the 60th percentile: classes 4 and 5
    c13 = score.lte(cuts[2])
    coll = ee.ImageCollection("GOOGLE/Research/open-buildings-temporal/v1")

    def year_img(y):
        a, b = ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y, 12, 31)
        m = coll.filterDate(a, b).mosaic()
        return m.select(["building_presence", "building_fractional_count"])
    teh = json.load(open(os.path.join(_ROOT, "data", "layyah_tehsils.geojson"), encoding="utf-8"))
    out = {}
    for f in teh["features"]:
        name = f["properties"]["adm3_name"]
        if name == "Choubara":
            continue
        g = ee.Geometry(f["geometry"]).intersection(layyah, 10)
        out[name] = {}
        groups = {"classes_4_5": c45, "classes_1_3": c13, "corridor": score.mask()}
        for gname, mask in groups.items():
            out[name][gname] = {}
            for y in (2016, 2023):
                im = year_img(y).updateMask(mask)
                r = im.reduceRegion(ee.Reducer.mean(), g, 30, maxPixels=1e11,
                                    bestEffort=True, tileScale=8).getInfo()
                out[name][gname][y] = {k: round(v, 5) if v is not None else None
                                       for k, v in r.items()}
        # whole tehsil (not just corridor)
        out[name]["tehsil"] = {}
        for y in (2016, 2023):
            r = year_img(y).reduceRegion(ee.Reducer.mean(), g, 30, maxPixels=1e11,
                                         bestEffort=True, tileScale=8).getInfo()
            out[name]["tehsil"][y] = {k: round(v, 5) if v is not None else None
                                      for k, v in r.items()}
    return out


def main():
    F.init_ee()
    layyah = F.district(F.TEST_DISTRICT)
    flood = flood_images()
    cached("points.json", lambda: part_points(layyah, flood))
    cached("per_year.json", lambda: part_per_year(layyah, flood))
    cached("bands.json", lambda: part_bands(layyah, flood))
    cached("buildings.json", lambda: part_buildings(layyah))
    print("done")


if __name__ == "__main__":
    main()

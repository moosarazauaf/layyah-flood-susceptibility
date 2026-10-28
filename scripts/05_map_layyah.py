"""
Susceptibility map, validation by area, and exposure for Layyah.

The final model is the logistic regression (all ten predictors), chosen on
training-only leave-one-district-out AUC (mean 0.912, worst district 0.856;
outputs/diagnostics.json). Its exact coefficients and scaling are applied
in Earth Engine, so the map is the fitted model, not an approximation, and
exposure is counted on the same grid. Before anything is trusted, the Earth
Engine scores are checked against the local model at the test points.

Susceptibility classes are quintiles of the score over Layyah's corridor:
class 5 is the most susceptible fifth of the land. Scores are relative
rankings, not calibrated probabilities (the model was trained on balanced
classes).

Outputs
  outputs/layyah_map_stats.json         check, class areas, observed floods
                                        per class, exposure per tehsil/class
  outputs/layyah_susceptibility_30m.tif score x 100 (uint8, 255 = outside)
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

import ee
import joblib
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import fs_common as F

OUT = os.path.join(_ROOT, "outputs")
SCALE = 30


def ee_logistic(model):
    sc, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    pred = F.predictors()
    z = (pred.subtract(ee.Image.constant(list(sc.mean_)))
         .divide(ee.Image.constant(list(sc.scale_)))
         .multiply(ee.Image.constant(list(lr.coef_[0])))
         .reduce(ee.Reducer.sum()).add(float(lr.intercept_[0])))
    return z.multiply(-1).exp().add(1).pow(-1).rename("score")


def area_by_class(cls, mask, geom):
    img = ee.Image.pixelArea().divide(1e6).updateMask(mask).addBands(cls)
    r = img.reduceRegion(ee.Reducer.sum().group(1, "cls"), geom, SCALE,
                         maxPixels=1e11, bestEffort=True, tileScale=8).getInfo()
    return {int(g["cls"]): round(g["sum"], 2) for g in r["groups"]}


def sum_by_class(cls, value, geom, scale):
    img = value.addBands(cls)
    r = img.reduceRegion(ee.Reducer.sum().group(1, "cls"), geom, scale,
                         maxPixels=1e11, bestEffort=True, tileScale=8).getInfo()
    return {int(g["cls"]): round(g["sum"], 1) for g in r["groups"]}


def flood_any(years, region):
    return ee.ImageCollection([F.flood_year(y, region)[0] for y in years]).max()


def fetch(url, path, tries=4):
    for a in range(1, tries + 1):
        try:
            urllib.request.urlretrieve(url, path); return
        except (urllib.error.HTTPError, OSError) as e:
            print(f"  retry {a}: {str(e)[:150]}"); time.sleep(15 * a)
    raise SystemExit(f"download failed: {path}")


def main():
    F.init_ee()
    model = joblib.load(os.path.join(OUT, "model_logistic.joblib"))
    layyah = F.district(F.TEST_DISTRICT)
    keep = F.corridor_mask().And(F.permanent_water().Not())
    score = ee_logistic(model).updateMask(keep).clip(layyah)
    stats = {}

    # 1. the Earth Engine model must reproduce the local one
    pts = json.load(open(os.path.join(OUT, "samples", "test_spatial_Layyah.json"), encoding="utf-8"))[:300]
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(p["lon"], p["lat"]), {"i": i})
                               for i, p in enumerate(pts)])
    ee_s = {f["properties"]["i"]: f["properties"].get("score")
            for f in score.sampleRegions(fc, scale=SCALE).getInfo()["features"]}
    X = np.array([[p[k] for k in F.PREDICTORS] for p in pts])
    local = model.predict_proba(X)[:, 1]
    diffs = [abs(ee_s[i] - local[i]) for i in ee_s if ee_s[i] is not None]
    stats["ee_vs_local"] = {"n": len(diffs), "max_abs_diff": round(max(diffs), 4),
                            "mean_abs_diff": round(float(np.mean(diffs)), 5)}
    print("Earth Engine vs local model at test points:", stats["ee_vs_local"])
    if max(diffs) > 0.02:
        raise SystemExit("Earth Engine model does not reproduce the local model")

    # 2. quintile classes over the corridor
    q = score.reduceRegion(ee.Reducer.percentile([20, 40, 60, 80]), layyah, SCALE,
                           maxPixels=1e11, bestEffort=True, tileScale=8).getInfo()
    cuts = [q[f"score_p{p}"] for p in (20, 40, 60, 80)]
    stats["quintile_cuts"] = [round(c, 4) for c in cuts]
    cls = (ee.Image(1).where(score.gt(cuts[0]), 2).where(score.gt(cuts[1]), 3)
           .where(score.gt(cuts[2]), 4).where(score.gt(cuts[3]), 5)
           .updateMask(score.mask()).rename("cls"))
    print("quintile cuts:", stats["quintile_cuts"])

    # 3. observed flooding by class, by area (policy-readable validation)
    f0019 = flood_any(F.TRAIN_YEARS, layyah)
    f2021 = flood_any(F.TEMPORAL_YEARS, layyah)
    frec = flood_any(F.RECENT_YEARS, layyah)
    stats["area_km2"] = {
        "all": area_by_class(cls, ee.Image(1), layyah),
        "flooded_2000_2019": area_by_class(cls, f0019, layyah),
        "first_flooded_2020_2021": area_by_class(cls, f2021.And(f0019.Not()), layyah),
        "first_flooded_2022_2025_suppl": area_by_class(
            cls, frec.And(f0019.Or(f2021).Not()), layyah)}
    for k, v in stats["area_km2"].items():
        tot = sum(v.values()) or 1
        print(f"  {k:<30}", " ".join(f"c{c} {100 * v.get(c, 0) / tot:5.1f}%" for c in range(1, 6)),
              f"| total {tot:.1f} km2")

    # 4. exposure by tehsil and class: people, built-up, cropland
    stats["exposure"] = {}
    teh = json.load(open(os.path.join(_ROOT, "data", "layyah_tehsils.geojson"), encoding="utf-8"))
    for f in teh["features"]:
        name = f["properties"]["adm3_name"]
        g = ee.Geometry(f["geometry"]).intersection(layyah, 10)
        e = {}
        for yr in (2000, 2010, 2020):
            e[f"pop_{yr}"] = sum_by_class(cls, ee.Image(f"JRC/GHSL/P2023A/GHS_POP/{yr}")
                                          .select("population_count"), g, 100)
            e[f"built_km2_{yr}"] = {k: round(v / 1e6, 3) for k, v in sum_by_class(
                cls, ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{yr}").select("built_surface"),
                g, 100).items()}
        e["cropland_km2"] = area_by_class(
            cls, ee.Image("ESA/WorldCover/v200/2021").select("Map").eq(40), g)
        e["area_km2"] = area_by_class(cls, ee.Image(1), g)
        stats["exposure"][name] = e
        hi = lambda d: sum(d.get(c, 0) for c in (4, 5))
        print(f"  {name:<16} classes 4-5: area {hi(e['area_km2']):6.1f} km2 | people "
              f"2000 {hi(e['pop_2000']):8,.0f} 2020 {hi(e['pop_2020']):8,.0f} | "
              f"cropland {hi(e['cropland_km2']):6.1f} km2")
    json.dump(stats, open(os.path.join(OUT, "layyah_map_stats.json"), "w", encoding="utf-8"), indent=2)
    print("wrote outputs/layyah_map_stats.json")

    # 5. score raster for the maps
    img = score.multiply(100).round().unmask(255).toUint8()
    url = img.getDownloadURL({"region": layyah.bounds(10), "scale": SCALE,
                              "crs": "EPSG:4326", "format": "GEO_TIFF"})
    path = os.path.join(OUT, "layyah_susceptibility_30m.tif")
    fetch(url, path)
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

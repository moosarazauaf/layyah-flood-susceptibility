"""
Sample points with flood labels and predictor values, per district.

Inside the 15 km Indus corridor, excluding JRC permanent water:

  cls (2000-2019, used for training and for Layyah's spatial test)
      1  flooded in at least one monsoon, 2000-2019
      0  never flooded 2000-2019, and observed in >= MIN_OBS_YEARS of 20
         (a pixel that was never seen cannot be called dry)

  cls_t (Layyah only, the temporal test, JRC labels throughout)
      1  first flooded 2020-2021: never flooded 2000-2019
      0  never flooded 2000-2021, observed as above
      The model, trained on 2000-2019 labels from other districts, has seen
      neither the place nor the period.

  cls_r (Layyah only, supplementary, 2022-2025)
      1  first flooded 2022-2025 (Landsat 8/9 continuation labels): never
         flooded 2000-2021 in the JRC record
      0  never flooded 2000-2025
      Supplementary because the continuation labels agree with JRC at only
      IoU 0.48-0.57 where both exist.

Negatives in the training districts are "not flooded 2000-2019" even where
the pixel floods later: that is all an analyst in 2019 could have known.

One file per district in outputs/samples/; finished districts are skipped,
so an interrupted run resumes.
"""
import json
import os
import sys

import ee

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import fs_common as F

OUT = os.path.join(_ROOT, "outputs", "samples")
N_TRAIN = 800        # per class per training district
N_TEST = 1000        # per class per Layyah test set
SEED = 42


def labels(region):
    def any_flood(years):
        f, o = zip(*(F.flood_year(y, region) for y in years))
        return ee.ImageCollection(list(f)).max(), ee.ImageCollection(list(o)).sum()
    f0019, obs0019 = any_flood(F.TRAIN_YEARS)
    f2021, _ = any_flood(F.TEMPORAL_YEARS)
    frecent, _ = any_flood(F.RECENT_YEARS)
    keep = F.corridor_mask().And(F.permanent_water().Not())
    seen = obs0019.gte(F.MIN_OBS_YEARS)
    cls = (ee.Image(0).where(f0019, 1)
           .updateMask(keep.And(seen.Or(f0019))).rename("cls"))
    # "first flooded after 2019" is only known for a pixel seen dry before,
    # so both classes of the temporal sets require the 2000-2019 record
    cls_t = (ee.Image(0).where(f2021, 1)
             .updateMask(keep.And(seen).And(f0019.Not())).rename("cls_t"))
    prior = f0019.Or(f2021)
    cls_r = (ee.Image(0).where(frecent, 1)
             .updateMask(keep.And(seen).And(prior.Not())).rename("cls_r"))
    return cls.toByte(), cls_t.toByte(), cls_r.toByte()


def sample(img, band, region, n):
    fc = img.stratifiedSample(numPoints=n, classBand=band, region=region,
                              scale=30, seed=SEED, geometries=True,
                              tileScale=8, dropNulls=True)
    return fc.getInfo()["features"]


def save(name, rows):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        json.dump(rows, fh)


def to_rows(features, label_band):
    rows = []
    for f in features:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        rows.append({"lon": lon, "lat": lat, "label": p[label_band],
                     **{k: p.get(k) for k in F.PREDICTORS}})
    return rows


def main():
    F.init_ee()
    os.makedirs(OUT, exist_ok=True)
    pred = F.predictors()
    jobs = [(d, "train") for d in F.TRAIN_DISTRICTS] + [(F.TEST_DISTRICT, "test_spatial")]
    # the temporal test depends on the 2022+ label method; --temporal once
    # that method has been fixed by scripts/01b_label_sensor_test.py
    if "--temporal" in sys.argv:
        jobs += [(F.TEST_DISTRICT, "test_temporal"), (F.TEST_DISTRICT, "test_recent")]
    for name, kind in jobs:
        fname = f"{kind}_{name.replace(' District', '').replace(' ', '_')}.json"
        if os.path.isfile(os.path.join(OUT, fname)):
            print(f"{fname}: already done"); continue
        region = F.district(name)
        cls, cls_t, cls_r = labels(region)
        band, lab = {"test_temporal": ("cls_t", cls_t),
                     "test_recent": ("cls_r", cls_r)}.get(kind, ("cls", cls))
        n = N_TEST if kind.startswith("test") else N_TRAIN
        feats = sample(pred.addBands(lab), band, region, n)
        rows = to_rows(feats, band)
        save(fname, rows)
        pos = sum(r["label"] == 1 for r in rows)
        print(f"{fname}: {len(rows)} points ({pos} flooded, {len(rows) - pos} not)")


if __name__ == "__main__":
    main()

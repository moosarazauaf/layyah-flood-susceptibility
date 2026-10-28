"""
Rasters for the maps.

  outputs/context_250m.tif        bands: corridor (0/1), indus (0/1), over the
                                  seven districts; display only
  outputs/inventory_strips/*.tif  Layyah corridor at 30 m, bands:
                                  years_flooded_2000_2019 (0-20; 255 outside),
                                  first_flooded_2020_2021 (0/1)
Inventory requests that fail are split into quarters and retried.
"""
import os
import sys
import time
import urllib.error
import urllib.request

import ee

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
import fs_common as F

OUT = os.path.join(_ROOT, "outputs")
STRIPS, MAX_DEPTH = 4, 2


def fetch(url, path, tries=2):
    for a in range(1, tries + 1):
        try:
            urllib.request.urlretrieve(url, path); return True
        except (urllib.error.HTTPError, OSError) as e:
            print(f"  retry {a}: {str(e)[:120]}"); time.sleep(15 * a)
    return False


def download_split(img, x0, x1, lo, hi, folder, tag, depth=0):
    path = os.path.join(folder, f"strip_{tag}.tif")
    if os.path.isfile(path):
        return
    box = ee.Geometry.Rectangle([x0, lo, x1, hi], None, False)
    url = img.getDownloadURL({"region": box, "scale": 30, "crs": "EPSG:4326",
                              "format": "GEO_TIFF"})
    if fetch(url, path):
        print(f"strip {tag}: ok"); return
    if depth >= MAX_DEPTH:
        raise SystemExit(f"strip {tag} still fails at depth {depth}")
    for j in range(4):
        download_split(img, x0, x1, lo + (hi - lo) * j / 4, lo + (hi - lo) * (j + 1) / 4,
                       folder, f"{tag}-{j}", depth + 1)


def main():
    F.init_ee()
    region = F.all_districts()
    ctx = os.path.join(OUT, "context_250m.tif")
    if not os.path.isfile(ctx):
        # MERIT's coarse pyramid levels average upstream area, so a one-pixel
        # river line falls below the threshold at 250 m and vanishes. Find
        # the river at native 90 m, then keep the maximum when coarsening.
        line = (F.indus().unmask(0).reproject(crs="EPSG:4326", scale=90)
                .reduceResolution(ee.Reducer.max(), maxPixels=64)
                .reproject(crs="EPSG:4326", scale=250))
        img = ee.Image.cat([F.corridor_mask().rename("corridor"),
                            line.rename("indus")]).toUint8().clip(region)
        url = img.getDownloadURL({"region": region.bounds(100), "scale": 250,
                                  "crs": "EPSG:4326", "format": "GEO_TIFF"})
        print("context:", "ok" if fetch(url, ctx, 3) else "FAILED")

    layyah = F.district(F.TEST_DISTRICT)
    keep = F.corridor_mask().And(F.permanent_water().Not())
    f_tr = [F.jrc_flood_year(y)[0] for y in F.TRAIN_YEARS]
    years = ee.ImageCollection(f_tr).sum()
    first = ee.ImageCollection([F.jrc_flood_year(y)[0] for y in F.TEMPORAL_YEARS]).max() \
        .And(years.eq(0))
    img = (ee.Image.cat([years.rename("years_flooded_2000_2019"),
                         first.rename("first_flooded_2020_2021")])
           .updateMask(keep).clip(layyah).unmask(255).toUint8())
    folder = os.path.join(OUT, "inventory_strips"); os.makedirs(folder, exist_ok=True)
    b = layyah.bounds(10).coordinates().get(0).getInfo()
    x0, x1 = min(p[0] for p in b), max(p[0] for p in b)
    y0, y1 = min(p[1] for p in b), max(p[1] for p in b)
    for i in range(STRIPS):
        download_split(img, x0, x1, y0 + (y1 - y0) * i / STRIPS,
                       y0 + (y1 - y0) * (i + 1) / STRIPS, folder, str(i))
    print("done")


if __name__ == "__main__":
    main()

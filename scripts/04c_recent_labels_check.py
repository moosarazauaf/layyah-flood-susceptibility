"""
Where is the 2022-2025 'first flooded' land? Local; no Earth Engine.

The supplementary 2022-2025 labels report 26 times more new flooding than
2020-2021 and the model places most of it in low classes. If those points
sit far from the Indus on vegetated farmland, unlike every other flood set,
they are more likely field water (irrigation or ponding) picked up by the
continuation classifier than river floods. Writes outputs/recent_labels_check.json.
"""
import importlib.util
import json
import os

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("te", os.path.join(_ROOT, "scripts", "03_train_evaluate.py"))
te = importlib.util.module_from_spec(spec); spec.loader.exec_module(te)

out = {}
sets = {"floods 2000-2019 (Layyah)": te.load("test_spatial_*.json"),
        "first flooded 2020-2021": te.load("test_temporal_*.json"),
        "first flooded 2022-2025 (suppl.)": te.load("test_recent_*.json")}
print(f"{'flooded points':<34}{'dist to Indus p50 (km)':>24}{'dry NDVI p50':>14}{'within 3 km':>13}")
for name, d in sets.items():
    f = d[d.label == 1]
    row = {"dist_indus_p50_km": round(float(f.dist_indus.median()) / 1000, 2),
           "ndvi_dry_p50": round(float(f.ndvi_dry.median()), 3),
           "share_within_3km": round(float((f.dist_indus <= 3000).mean()), 3)}
    out[name] = row
    print(f"{name:<34}{row['dist_indus_p50_km']:>24}{row['ndvi_dry_p50']:>14}{row['share_within_3km']:>13.1%}")
json.dump(out, open(os.path.join(_ROOT, "outputs", "recent_labels_check.json"), "w"), indent=2)
print("wrote outputs/recent_labels_check.json")

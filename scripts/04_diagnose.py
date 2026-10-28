"""
Diagnostics before choosing a final model. Local; no Earth Engine.

1. Leave-one-district-out AUC per district, per model.
2. Which predictors each tree model relies on (permutation importance on
   Layyah's spatial and temporal tests, so it measures what transfers).
3. HAND: flooded vs never-flooded distributions, and HAND against distance
   to the Indus, to see why HAND alone scores at or below chance here.
4. Ablation, judged on training-only leave-one-district-out (choosing on
   the Layyah tests would leak them into model selection):
     terrain    dist_indus, dist_drain, elev, slope, twi, hand
     +indices   terrain + ndvi_dry, ndbi_dry
     all        + clay, rain
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from fs_common import PREDICTORS  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "te", os.path.join(_ROOT, "scripts", "03_train_evaluate.py"))
te = importlib.util.module_from_spec(spec); spec.loader.exec_module(te)

OUT = os.path.join(_ROOT, "outputs")
SETS = {"terrain": ["dist_indus", "dist_drain", "elev", "slope", "twi", "hand"],
        "terrain+indices": ["dist_indus", "dist_drain", "elev", "slope", "twi", "hand",
                            "ndvi_dry", "ndbi_dry"],
        "all": PREDICTORS}


def xgb():
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                         random_state=42, n_jobs=-1)


def lodo(tr, cols):
    aucs = {}
    for d in sorted(tr.district.unique()):
        a, b = tr.district != d, tr.district == d
        m = xgb().fit(tr.loc[a, cols], tr.label[a])
        aucs[d] = round(roc_auc_score(tr.label[b], m.predict_proba(tr.loc[b, cols])[:, 1]), 3)
    return aucs


def main():
    tr, sp, tt = te.load("train_*.json"), te.load("test_spatial_*.json"), te.load("test_temporal_*.json")
    rc = te.load("test_recent_*.json")
    res = json.load(open(os.path.join(OUT, "model_results.json"), encoding="utf-8"))
    out = {}

    print("1. leave-one-district-out AUC")
    for name, r in res["models"].items():
        print(f"   {name:<14}", "  ".join(f"{d[:10]} {v['auc_roc']:.3f}" for d, v in r["lodo"].items()))

    print("\n2. permutation importance (drop in AUC when a predictor is shuffled)")
    out["importance"] = {}
    for name in ("xgboost", "random_forest"):
        m = joblib.load(os.path.join(OUT, f"model_{name}.joblib"))
        out["importance"][name] = {}
        for tag, d in (("spatial", sp), ("temporal", tt)):
            pi = permutation_importance(m, d[PREDICTORS], d.label, scoring="roc_auc",
                                        n_repeats=10, random_state=42, n_jobs=-1)
            imp = dict(sorted(zip(PREDICTORS, pi.importances_mean.round(3)),
                              key=lambda kv: -kv[1]))
            out["importance"][name][tag] = imp
            print(f"   {name:<14} {tag:<9}", "  ".join(f"{k} {v:+.3f}" for k, v in list(imp.items())[:6]))

    print("\n3. HAND diagnostic (training districts)")
    q = lambda s: [round(float(x), 2) for x in np.percentile(s, [10, 50, 90])]
    f, n = tr[tr.label == 1], tr[tr.label == 0]
    out["hand"] = {"flooded_p10_50_90": q(f.hand), "dry_p10_50_90": q(n.hand),
                   "dist_indus_flooded_p50_m": round(float(f.dist_indus.median())),
                   "dist_indus_dry_p50_m": round(float(n.dist_indus.median())),
                   "share_hand_zero_flooded": round(float((f.hand <= 0.5).mean()), 3),
                   "share_hand_zero_dry": round(float((n.hand <= 0.5).mean()), 3),
                   "spearman_hand_vs_dist_indus": round(float(
                       tr[["hand", "dist_indus"]].corr("spearman").iloc[0, 1]), 3)}
    for k, v in out["hand"].items():
        print(f"   {k}: {v}")

    print("\n4. ablation, XGBoost, leave-one-district-out (training only)")
    out["ablation"] = {}
    for tag, cols in SETS.items():
        a = lodo(tr, cols)
        m = xgb().fit(tr[cols], tr.label)
        row = {"lodo": a, "lodo_mean": round(float(np.mean(list(a.values()))), 3),
               "lodo_min": min(a.values()),
               "layyah_spatial": round(roc_auc_score(sp.label, m.predict_proba(sp[cols])[:, 1]), 3),
               "layyah_temporal": round(roc_auc_score(tt.label, m.predict_proba(tt[cols])[:, 1]), 3),
               "layyah_recent_suppl": round(roc_auc_score(rc.label, m.predict_proba(rc[cols])[:, 1]), 3)}
        out["ablation"][tag] = row
        print(f"   {tag:<16} LODO mean {row['lodo_mean']:.3f} (min {row['lodo_min']:.3f}) | "
              f"Layyah spatial {row['layyah_spatial']:.3f} temporal {row['layyah_temporal']:.3f} "
              f"2022-25 {row['layyah_recent_suppl']:.3f}")
    json.dump(out, open(os.path.join(OUT, "diagnostics.json"), "w", encoding="utf-8"), indent=2)
    print("\nwrote outputs/diagnostics.json")


if __name__ == "__main__":
    main()

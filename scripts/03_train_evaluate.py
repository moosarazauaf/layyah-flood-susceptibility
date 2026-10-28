"""
Train and evaluate flood-susceptibility models. Local; no Earth Engine.

Models: logistic regression (standardised), random forest, XGBoost.

Evaluation, none of it on randomly split points (neighbouring pixels are
near-copies, so random splits inflate accuracy):
  1. spatial-block CV inside the six training districts (10 km blocks)
  2. leave-one-district-out over the six training districts
  3. Layyah spatial test: train on all six, predict Layyah 2000-2019 floods
  4. Layyah temporal test: same model, predict land first flooded 2020-2021
  5. supplementary: land first flooded 2022-2025 (continuation labels,
     IoU 0.48-0.57 with JRC where both exist, so reported separately)

Two trivial baselines are scored on the same tests: distance to the Indus
alone, and HAND alone. If the models cannot beat them, the honest result
is that a simple distance rule serves planners equally well.

Writes outputs/model_results.json and outputs/model_<name>.joblib.
"""
import glob
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from fs_common import PREDICTORS  # noqa: E402  (pure constants; no EE call)

SAMPLES = os.path.join(_ROOT, "outputs", "samples")
OUT = os.path.join(_ROOT, "outputs")
BLOCK_DEG = 0.1          # ~10 km spatial blocks
SEED = 42


def models():
    return {
        "logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=SEED,
            n_jobs=-1),
    }


def load(pattern):
    frames = []
    paths = sorted(glob.glob(os.path.join(SAMPLES, pattern)))
    if not paths:
        print(f"  {pattern}: not sampled yet, skipped")
        return None
    for path in paths:
        df = pd.DataFrame(json.load(open(path, encoding="utf-8")))
        df["district"] = os.path.basename(path).split("_", 1)[1].rsplit(".", 1)[0]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.dropna(subset=PREDICTORS)
    if len(df) < before:
        print(f"  {pattern}: dropped {before - len(df)} rows with missing predictors")
    return df


def scores(y, p):
    return {"auc_roc": round(roc_auc_score(y, p), 3),
            "auc_pr": round(average_precision_score(y, p), 3), "n": int(len(y))}


def baselines(df):
    if df is None:
        return None
    # low distance and low HAND mean more flood-prone, so negate
    return {"baseline_dist_indus": scores(df.label, -df.dist_indus),
            "baseline_hand": scores(df.label, -df.hand)}


def auc(r):
    return "n/a" if r is None else f"{r['auc_roc']:.3f}"


def main():
    tr = load("train_*.json")
    sp = load("test_spatial_*.json")
    tt = load("test_temporal_*.json")
    rc = load("test_recent_*.json")
    for name, d in (("train", tr), ("Layyah spatial", sp), ("Layyah temporal", tt),
                    ("Layyah 2022-25", rc)):
        if d is not None:
            print(f"{name:<16} {len(d):5d} points, {int(d.label.sum()):5d} flooded")
    X, y = tr[PREDICTORS].values, tr.label.values
    blocks = (np.floor(tr.lon / BLOCK_DEG).astype(int) * 10000
              + np.floor(tr.lat / BLOCK_DEG).astype(int))

    n = lambda d: None if d is None else len(d)
    res = {"n": {"train": len(tr), "test_spatial": len(sp), "test_temporal": n(tt),
                 "test_recent_supplementary": n(rc)},
           "baselines": {"layyah_spatial": baselines(sp),
                         "layyah_temporal": baselines(tt),
                         "layyah_recent_supplementary": baselines(rc),
                         "train": baselines(tr)},
           "models": {}}
    for name, mk in models().items():
        r = {}
        # 1. spatial-block CV
        oof = np.zeros(len(y))
        for a, b in GroupKFold(n_splits=5).split(X, y, blocks):
            m = models()[name]; m.fit(X[a], y[a]); oof[b] = m.predict_proba(X[b])[:, 1]
        r["block_cv"] = scores(y, oof)
        # 2. leave-one-district-out
        r["lodo"] = {}
        for d in sorted(tr.district.unique()):
            a, b = tr.district != d, tr.district == d
            m = models()[name]; m.fit(X[a.values], y[a.values])
            r["lodo"][d] = scores(y[b.values], m.predict_proba(X[b.values])[:, 1])
        # 3./4. train on all six, test on Layyah
        m = models()[name]; m.fit(X, y)
        r["layyah_spatial"] = scores(sp.label, m.predict_proba(sp[PREDICTORS].values)[:, 1])
        for key, d in (("layyah_temporal", tt), ("layyah_recent_supplementary", rc)):
            r[key] = None if d is None else scores(
                d.label, m.predict_proba(d[PREDICTORS].values)[:, 1])
        joblib.dump(m, os.path.join(OUT, f"model_{name}.joblib"))
        res["models"][name] = r
        lodo = [v["auc_roc"] for v in r["lodo"].values()]
        print(f"{name:<14} block-CV AUC {r['block_cv']['auc_roc']:.3f} | "
              f"LODO AUC {min(lodo):.3f}-{max(lodo):.3f} | "
              f"Layyah spatial {r['layyah_spatial']['auc_roc']:.3f} | "
              f"Layyah temporal {auc(r['layyah_temporal'])} | "
              f"2022-25 (suppl.) {auc(r['layyah_recent_supplementary'])}")
    for k, v in res["baselines"].items():
        if v is None:
            continue
        print(f"baselines on {k:<16} dist_indus AUC {v['baseline_dist_indus']['auc_roc']:.3f}"
              f" | HAND AUC {v['baseline_hand']['auc_roc']:.3f}")
    json.dump(res, open(os.path.join(OUT, "model_results.json"), "w", encoding="utf-8"),
              indent=2)
    print("wrote outputs/model_results.json")


if __name__ == "__main__":
    main()

"""
Local analyses requested in the peer review. No Earth Engine.

 A  spatial block-bootstrap 95% intervals (0.1 degree blocks) for every Layyah
    AUC and for the logistic-minus-distance difference            (R1-2)
 B  ablation for the FINAL model (logistic): all predictors, without the
    dry-season indices, terrain only                              (R1-4)
 C  subset evaluations that need outputs/revision/points.json:
      - cropland points only (ESA WorldCover class 40)            (R1-5)
      - flooded points that flooded in a year other than 2010     (R2-1)
      - points north of 30.76 N, away from Taunsa Barrage pondage (R2-2)
 D  AUC-PR and per-model summaries                                 (R1-7)

Writes outputs/revision/local.json.
"""
import importlib.util
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from fs_common import PREDICTORS  # noqa: E402

spec = importlib.util.spec_from_file_location("te", os.path.join(_ROOT, "scripts", "03_train_evaluate.py"))
te = importlib.util.module_from_spec(spec); spec.loader.exec_module(te)

OUT = os.path.join(_ROOT, "outputs")
REV = os.path.join(OUT, "revision")
os.makedirs(REV, exist_ok=True)
RNG = np.random.default_rng(42)
B = 2000
MODELS = ["logistic", "random_forest", "xgboost"]
SOUTH_CUT_LATS = (30.76, 30.90)   # 27 and 43 km north of the Taunsa Barrage (30.513 N); Ramsar pond centre 30.70 N


def blocks(df):
    return (np.floor(df.lon / 0.1).astype(int) * 10000 + np.floor(df.lat / 0.1).astype(int)).values


def boot_auc(y, scores, groups, reps=B):
    """scores: dict name -> array. Returns percentile intervals and, for each
    name, the difference to 'distance' when present."""
    ub = np.unique(groups)
    idx_by = {b: np.nonzero(groups == b)[0] for b in ub}
    names = list(scores)
    draws = {n: [] for n in names}
    diffs = {n: [] for n in names if n != "distance"}
    for _ in range(reps):
        pick = RNG.choice(ub, size=len(ub), replace=True)
        idx = np.concatenate([idx_by[b] for b in pick])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        a = {n: roc_auc_score(yy, scores[n][idx]) for n in names}
        for n in names:
            draws[n].append(a[n])
        for n in diffs:
            diffs[n].append(a[n] - a["distance"])
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return ({n: ci(v) for n, v in draws.items()},
            {n: {"mean": round(float(np.mean(v)), 3), "ci": ci(v),
                 "share_positive": round(float((np.array(v) > 0).mean()), 3)}
             for n, v in diffs.items()})


def model_scores(df):
    s = {m: joblib.load(os.path.join(OUT, f"model_{m}.joblib")).predict_proba(df[PREDICTORS])[:, 1]
         for m in MODELS}
    s["distance"] = -df.dist_indus.values
    s["hand"] = -df.hand.values
    return s


def main():
    sp, tt, rc = te.load("test_spatial_*.json"), te.load("test_temporal_*.json"), te.load("test_recent_*.json")
    tr = te.load("train_*.json")
    res = {"A_bootstrap": {}, "B_ablation": {}, "C_subsets": {}, "D_auc_pr": {}}

    print("A. block bootstrap, 95% intervals")
    for tag, d in (("2000-2019", sp), ("first_flooded_2020-2021", tt), ("2022-2025_suppl", rc)):
        y, s = d.label.values, model_scores(d)
        ci, diff = boot_auc(y, s, blocks(d))
        res["A_bootstrap"][tag] = {"n_blocks": int(len(np.unique(blocks(d)))),
                                    "auc": {n: round(float(roc_auc_score(y, s[n])), 3) for n in s},
                                    "ci": ci, "minus_distance": diff}
        r = res["A_bootstrap"][tag]
        print(f"  {tag}: {r['n_blocks']} blocks")
        for n in ("logistic", "random_forest", "xgboost", "distance"):
            print(f"    {n:<14} {r['auc'][n]:.3f} {ci[n]}")
        for n, v in diff.items():
            if n != "hand":
                print(f"    {n:<14} minus distance {v['mean']:+.3f} {v['ci']}  (>0 in {v['share_positive']:.0%} of draws)")

    print("\nB. ablation of the final logistic model")
    sets = {"all ten predictors": PREDICTORS,
            "without dry-season NDVI and NDBI": [p for p in PREDICTORS if p not in ("ndvi_dry", "ndbi_dry")],
            "terrain, drainage and distance only": ["hand", "elev", "slope", "dist_indus", "dist_drain", "twi"],
            "distance to Indus only": ["dist_indus"]}
    for name, cols in sets.items():
        lodo = {}
        for dname in sorted(tr.district.unique()):
            a, b = tr.district != dname, tr.district == dname
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr.loc[a, cols], tr.label[a])
            lodo[dname] = roc_auc_score(tr.label[b], m.predict_proba(tr.loc[b, cols])[:, 1])
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr[cols], tr.label)
        row = {"lodo_mean": round(float(np.mean(list(lodo.values()))), 3),
               "lodo_worst": round(float(min(lodo.values())), 3),
               "layyah_2000_2019": round(float(roc_auc_score(sp.label, m.predict_proba(sp[cols])[:, 1])), 3),
               "layyah_2020_2021": round(float(roc_auc_score(tt.label, m.predict_proba(tt[cols])[:, 1])), 3)}
        res["B_ablation"][name] = row
        print(f"  {name:<38} LODO mean {row['lodo_mean']:.3f} worst {row['lodo_worst']:.3f} | "
              f"Layyah 2000-19 {row['layyah_2000_2019']:.3f}  2020-21 {row['layyah_2020_2021']:.3f}")

    print("\nD. AUC-PR (Layyah)")
    mr = json.load(open(os.path.join(OUT, "model_results.json"), encoding="utf-8"))
    for m, v in mr["models"].items():
        res["D_auc_pr"][m] = {"2000-2019": v["layyah_spatial"]["auc_pr"],
                              "2020-2021": v["layyah_temporal"]["auc_pr"]}
    res["D_auc_pr"]["distance"] = {"2000-2019": mr["baselines"]["layyah_spatial"]["baseline_dist_indus"]["auc_pr"],
                                   "2020-2021": mr["baselines"]["layyah_temporal"]["baseline_dist_indus"]["auc_pr"]}
    print("  ", json.dumps(res["D_auc_pr"]))

    pts_path = os.path.join(REV, "points.json")
    if os.path.isfile(pts_path):
        print("\nC. subset evaluations")
        pts = pd.DataFrame(json.load(open(pts_path, encoding="utf-8")))
        for tag, d in (("2000-2019", sp), ("first_flooded_2020-2021", tt)):
            key = "spatial" if tag == "2000-2019" else "temporal"
            p = pts[pts.set == key].sort_values("i").reset_index(drop=True)
            assert len(p) == len(d), "points.json does not align with the sample file"
            d = d.reset_index(drop=True).copy()
            for c in ("f2010", "n_excl2010", "wc"):
                d[c] = p[c].values
            s = model_scores(d)
            def sub(mask, name):
                dd, y = d[mask], d.label.values[mask]
                if len(np.unique(y)) < 2 or min((y == 0).sum(), (y == 1).sum()) < 30:
                    res["C_subsets"].setdefault(tag, {})[name] = {"n": int(mask.sum()), "note": "too few points"}
                    print(f"  {tag:<24} {name:<34} n={mask.sum()} (too few of one class)")
                    return
                sc = {n: v[mask.values] for n, v in s.items()}
                ci, diff = boot_auc(y, sc, blocks(dd), reps=1000)
                row = {"n": int(mask.sum()), "positives": int(y.sum()),
                       "auc": {n: round(float(roc_auc_score(y, sc[n])), 3) for n in ("logistic", "xgboost", "distance")},
                       "logistic_ci": ci["logistic"], "distance_ci": ci["distance"],
                       "logistic_minus_distance": diff["logistic"]}
                res["C_subsets"].setdefault(tag, {})[name] = row
                print(f"  {tag:<24} {name:<34} n={row['n']:4d} pos={row['positives']:4d} "
                      f"logistic {row['auc']['logistic']:.3f} {ci['logistic']} | distance {row['auc']['distance']:.3f} "
                      f"| diff {diff['logistic']['mean']:+.3f} {diff['logistic']['ci']}")
            allm = pd.Series(True, index=d.index)
            sub(allm, "all points")
            sub(d.wc == 40, "cropland only (WorldCover 40)")
            if tag == "2000-2019":
                pos_ok = (d.label == 0) | (d.n_excl2010 >= 1)
                sub(pos_ok, "positives flooded in a year other than 2010")
            for cut in SOUTH_CUT_LATS:
                sub(d.lat >= cut, f"north of {cut} N (away from Taunsa pond)")
        wc = pts.wc.value_counts(normalize=True).round(3).to_dict()
        res["C_subsets"]["worldcover_share_all_points"] = {str(int(k)): v for k, v in wc.items() if pd.notna(k)}
        print("  WorldCover class shares over all test points:", res["C_subsets"]["worldcover_share_all_points"])
    else:
        print("\nC. skipped: outputs/revision/points.json not written yet")
    json.dump(res, open(os.path.join(REV, "local.json"), "w", encoding="utf-8"), indent=2)
    print("\nwrote outputs/revision/local.json")


if __name__ == "__main__":
    main()

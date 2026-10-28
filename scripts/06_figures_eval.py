"""
Evaluation figures, one file each at 600 dpi, 6.5 in (16.5 cm) wide, no
text under 9.5 pt. Local; no Earth Engine.

figA_roc_spatial.png    ROC on Layyah 2000-2019 floods: models and baselines
figB_roc_temporal.png   ROC on land first flooded 2020-2021
figC_importance.png     permutation importance of the final model
"""
import os
import sys
import importlib.util

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from fs_common import PREDICTORS  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "te", os.path.join(_ROOT, "scripts", "03_train_evaluate.py"))
te = importlib.util.module_from_spec(spec); spec.loader.exec_module(te)

OUT = os.path.join(_ROOT, "outputs")
FIG = os.path.join(_ROOT, "paper", "figures")
DPI, W, FS = 600, 6.5, 10
LABELS = {"hand": "Height above nearest drainage (HAND)", "elev": "Elevation",
          "slope": "Slope", "dist_indus": "Distance to the Indus",
          "dist_drain": "Distance to any drainage", "twi": "Topographic wetness index",
          "clay": "Soil clay content", "rain": "Mean monsoon rainfall",
          "ndvi_dry": "Dry-season NDVI", "ndbi_dry": "Dry-season NDBI"}
STYLE = {"logistic": ("Logistic regression (final model)", "#1a1a1a", "-", 2.2),
         "random_forest": ("Random forest", "#2166ac", "-", 1.3),
         "xgboost": ("XGBoost", "#1b7837", "-", 1.3)}


def roc_figure(df, fname):
    fig, ax = plt.subplots(figsize=(W, 4.6), layout="constrained")
    for name, (lab, col, ls, lw) in STYLE.items():
        m = joblib.load(os.path.join(OUT, f"model_{name}.joblib"))
        p = m.predict_proba(df[PREDICTORS])[:, 1]
        fpr, tpr, _ = roc_curve(df.label, p)
        ax.plot(fpr, tpr, ls, color=col, lw=lw,
                label=f"{lab} (AUC {roc_auc_score(df.label, p):.3f})")
    for col_name, lab, col in (("dist_indus", "Distance to the Indus alone", "#d95f02"),
                               ("hand", "HAND alone", "#999999")):
        s = -df[col_name]
        fpr, tpr, _ = roc_curve(df.label, s)
        ax.plot(fpr, tpr, "--", color=col, lw=1.5,
                label=f"{lab} (AUC {roc_auc_score(df.label, s):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="#bbbbbb", lw=1)
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.tick_params(labelsize=FS)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(FIG, fname), dpi=DPI, facecolor="white")
    plt.close(fig); print("wrote", fname)


def importance_figure(sp, tt):
    """Permutation importance of the final model on both Layyah tests.

    Not coefficients: elevation and rainfall are collinear (Spearman 0.93),
    so their coefficients are not individually interpretable. Importance
    asks what a planner needs: how much worse the map gets without it."""
    import json
    import numpy as np
    from sklearn.inspection import permutation_importance
    m = joblib.load(os.path.join(OUT, "model_logistic.joblib"))
    imp = {}
    for tag, d in (("spatial", sp), ("temporal", tt)):
        r = permutation_importance(m, d[PREDICTORS], d.label, scoring="roc_auc",
                                   n_repeats=20, random_state=42, n_jobs=-1)
        imp[tag] = dict(zip(PREDICTORS, r.importances_mean))
    json.dump({t: {k: round(v, 4) for k, v in d.items()} for t, d in imp.items()},
              open(os.path.join(OUT, "importance_final_model.json"), "w"), indent=2)
    order = sorted(PREDICTORS, key=lambda k: imp["spatial"][k] + imp["temporal"][k])
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(W, 4.4), layout="constrained")
    ax.barh(y + 0.2, [imp["spatial"][k] for k in order], 0.4, color="#1a1a1a",
            label="Layyah, floods 2000-2019")
    ax.barh(y - 0.2, [imp["temporal"][k] for k in order], 0.4, color="#d95f02",
            label="Layyah, first flooded 2020-2021")
    ax.set_yticks(y, [LABELS[k] for k in order])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Drop in AUC when the predictor is shuffled", fontsize=11)
    ax.tick_params(labelsize=FS)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(FIG, "figC_importance.png"), dpi=DPI, facecolor="white")
    plt.close(fig); print("wrote figC_importance.png")
    for k in reversed(order):
        print(f"  {k:<11} spatial {imp['spatial'][k]:+.3f}  temporal {imp['temporal'][k]:+.3f}")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    sp, tt = te.load("test_spatial_*.json"), te.load("test_temporal_*.json")
    roc_figure(sp, "figA_roc_spatial.png")
    roc_figure(tt, "figB_roc_temporal.png")
    importance_figure(sp, tt)

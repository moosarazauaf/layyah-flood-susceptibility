"""
Collinearity among predictors (training sample). Writes
outputs/collinearity.json. Variance inflation factor above 5 is a concern.
Finding: elevation and mean monsoon rainfall are strongly correlated
(Spearman ~0.93; both trend north-south along the river), so their
individual logistic coefficients are not interpretable. HAND is not
collinear (VIF ~2.4); its coefficient is a conditional effect.
"""
import json
import importlib.util, os, sys
import numpy as np
from sklearn.linear_model import LinearRegression
sys.path.insert(0, "src")
from fs_common import PREDICTORS
spec = importlib.util.spec_from_file_location("te", "scripts/03_train_evaluate.py")
te = importlib.util.module_from_spec(spec); spec.loader.exec_module(te)
tr = te.load("train_*.json")
X = tr[PREDICTORS]
print("variance inflation factor (>5 is a concern, >10 serious):")
vif = {}
for c in PREDICTORS:
    others = [o for o in PREDICTORS if o != c]
    r2 = LinearRegression().fit(X[others], X[c]).score(X[others], X[c])
    vif[c] = round(1 / (1 - r2), 2)
    print(f"  {c:<11} VIF {vif[c]:6.1f}")
cm = X.corr("spearman")
pairs = [(a, b, cm.loc[a, b]) for i, a in enumerate(PREDICTORS) for b in PREDICTORS[i + 1:]]
print("strongest pairwise Spearman correlations:")
for a, b, v in sorted(pairs, key=lambda t: -abs(t[2]))[:6]:
    print(f"  {a:<11} {b:<11} {v:+.2f}")
json.dump({"vif": vif, "top_spearman": [[a, b, round(float(v), 3)] for a, b, v in
           sorted(pairs, key=lambda t: -abs(t[2]))[:6]]},
          open("outputs/collinearity.json", "w", encoding="utf-8"), indent=2)
print("wrote outputs/collinearity.json")

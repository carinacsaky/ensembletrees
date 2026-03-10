"""
Standalone private insurance penetration predictor.
Requires only: joblib, numpy  (no DB, no geopandas).

Usage
-----
  python predict_private.py county.json
  python predict_private.py county.json --model output/private_model.pkl

The JSON file should contain a flat dict of county-level feature values, e.g.:

    {
        "n_locuinte_county": 371000,
        "urbn_type": 1,
        "mount_type": 1,
        "coast_type": 0,
        "lat": 46.80,
        "lon": 23.60,
        "pop_density": 93.5,
        "avg_waterway_dist_km": 1.2
    }

n_locuinte_county   : total housing stock in the county (log transform applied automatically)
urbn_type           : 1=urban agglomeration, 2=intermediate, 3=rural  (from NUTS3)
mount_type          : 1=mountain, 2=intermediate, 3=non-mountain  (from NUTS3)
coast_type          : 1=coastal, 2=intermediate, 3=non-coastal  (from NUTS3)
lat/lon             : county centroid coordinates  (from NUTS3)
pop_density         : housing units per km² of county area
avg_waterway_dist_km: mean distance from localities to nearest waterway (km)

Any feature not supplied defaults to 0.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL = Path(__file__).parent / "output" / "private_model.pkl"

parser = argparse.ArgumentParser(description="Predict private insurance penetration from a JSON feature file.")
parser.add_argument("features_file", type=Path, help="Path to JSON file with county feature values")
parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to private_model.pkl")
args = parser.parse_args()

if not args.features_file.exists():
    print(f"Error: features file not found: {args.features_file}")
    sys.exit(1)

if not args.model.exists():
    print(f"Error: model file not found: {args.model}")
    print("Run private_insurance.py first to train and save the model.")
    sys.exit(1)

with open(args.features_file) as f:
    features_dict = json.load(f)

# Accept n_locuinte_county as raw count and convert to log
n_locuinte = features_dict.pop("n_locuinte_county", None)
if n_locuinte is not None:
    features_dict["log_n_locuinte_county"] = float(np.log1p(n_locuinte))

bundle       = joblib.load(args.model)
bundle_models = bundle["models"]
feature_cols  = bundle["feature_cols"]

row    = {col: features_dict.get(col, 0) for col in feature_cols}
X_pred = pd.DataFrame([row], columns=feature_cols)

PERILS = ["flood", "earthquake", "fire"]

def _sigmoid(x):
    return float(1 / (1 + np.exp(-x)))

print("\n--- Input ---")
if n_locuinte is not None:
    print(f"  n_locuinte_county: {int(n_locuinte):,}")
for feat, val in features_dict.items():
    if feat != "log_n_locuinte_county":
        print(f"  {feat}: {val}")

print("\n--- Private insurance penetration prediction ---")
predictions = {}
for peril in PERILS:
    m         = bundle_models[peril]
    cv_scores = m["cv_scores"]

    pred_lgbm = _sigmoid(m["lgbm"].predict(X_pred)[0])
    pred_rf   = _sigmoid(m["rf"].predict(X_pred)[0])
    pred_xgb  = _sigmoid(m["xgb"].predict(X_pred)[0])

    total = sum(cv_scores.values()) if cv_scores else 0
    if total > 0:
        w = {k: v / total for k, v in cv_scores.items()}
        ensemble = pred_lgbm * w["LightGBM"] + pred_rf * w["RandomForest"] + pred_xgb * w["XGBoost"]
    else:
        ensemble = float(np.mean([pred_lgbm, pred_rf, pred_xgb]))

    predictions[peril] = ensemble
    print(f"  {peril.capitalize():<12}: {ensemble:.1%}  "
          f"(LightGBM {pred_lgbm:.1%} / RF {pred_rf:.1%} / XGBoost {pred_xgb:.1%})")

if n_locuinte is not None:
    print("\n--- Market sizing ---")
    for peril in PERILS:
        est = int(round(n_locuinte * predictions[peril]))
        print(f"  Estimated privately insured buildings ({peril}): {est:,}")

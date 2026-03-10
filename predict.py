"""
Standalone PAD coverage rate predictor.
Requires only: joblib, numpy, pandas  (no DB, no geopandas, no shap).

Usage
-----
  python predict.py location.json
  python predict.py location.json --model output/model.pkl

The JSON file should contain a flat dict of feature values, e.g.:

    {
        "tip_code": 1,
        "n_locuinte": 50000,
        "n_pad": 8000,
        "loc_lat": 46.77,
        "loc_lon": 23.59,
        "urbn_type": 1,
        "mount_type": 4,
        "coast_type": 2,
        "dist_to_waterway_km": 2.1,
        "n_buildings_flood": 18403,
        "n_buildings_earthquake": 18398,
        "n_buildings_fire": 18296,
        "n_active_perils": 3,
        "log_total_buildings": 10.917,
        "avg_insured_net_flood": 299385.68,
        "avg_insured_net_fire": 300049.17,
        "retention_ratio_flood": 0.6667,
        "retention_ratio_earthquake": 0.6667,
        "retention_ratio_fire": 0.6667
    }

tip_code:   1=Municipiu, 2=Oras, 3=Comuna
n_locuinte: raw housing count (log transform applied automatically)
n_pad:      actual insured count (optional — used to compute the opportunity gap)
Building features (n_buildings_*, avg_insured_net_*, retention_ratio_*):
            county-level — look up from data/cache/building_counts_ro.csv

Any feature not supplied defaults to 0.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL = Path(__file__).parent / "output" / "model.pkl"

parser = argparse.ArgumentParser(description="Predict PAD coverage rate from a JSON feature file.")
parser.add_argument("features_file", type=Path, help="Path to JSON file with feature values")
parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to model.pkl")
args = parser.parse_args()

if not args.features_file.exists():
    print(f"Error: features file not found: {args.features_file}")
    sys.exit(1)

if not args.model.exists():
    print(f"Error: model file not found: {args.model}")
    print("Run ensembletrees.py first to train and save the model.")
    sys.exit(1)

with open(args.features_file) as f:
    features_dict = json.load(f)

# Accept n_locuinte in plain form and convert to the log-transformed feature the model expects
n_locuinte        = features_dict.pop("n_locuinte", None)
n_pad             = features_dict.pop("n_pad", None)           # optional: actual insured count
current_rate      = features_dict.pop("current_coverage_rate", None)  # optional: actual rate (0–1)
if n_locuinte is not None:
    features_dict["log_n_locuinte"] = float(np.log1p(n_locuinte))
# Derive current_rate from n_pad if not supplied directly
if current_rate is None and n_pad is not None and n_locuinte:
    current_rate = n_pad / n_locuinte

bundle = joblib.load(args.model)
lgbm               = bundle["lgbm"]
rf                 = bundle["rf"]
xgb                = bundle["xgb"]
cv_scores          = bundle["cv_scores"]
feature_cols       = bundle["feature_cols"]
county_mean_lookup = bundle.get("county_mean_lookup", {})

# Auto-fill county_mean_coverage from the bundle when judet is provided
if "county_mean_coverage" not in features_dict and "judet" in features_dict:
    import unicodedata
    def _normalize(s):
        s = unicodedata.normalize("NFD", str(s))
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return s.strip().upper()
    key = _normalize(features_dict["judet"])
    features_dict["county_mean_coverage"] = county_mean_lookup.get(key, 0.0)
    if key not in county_mean_lookup:
        print(f"Warning: county '{features_dict['judet']}' not found in model — county_mean_coverage=0")

row    = {col: features_dict.get(col, 0) for col in feature_cols}
X_pred = pd.DataFrame([row])

def _sigmoid(x):
    return float(1 / (1 + np.exp(-x)))

pred_lgbm = _sigmoid(lgbm.predict(X_pred)[0])
pred_rf   = _sigmoid(rf.predict(X_pred)[0])
pred_xgb  = _sigmoid(xgb.predict(X_pred)[0])

total = sum(cv_scores.values()) if cv_scores else 0
if total > 0:
    w = {k: v / total for k, v in cv_scores.items()}
    ensemble = pred_lgbm * w["LightGBM"] + pred_rf * w["RandomForest"] + pred_xgb * w["XGBoost"]
else:
    ensemble = float(np.mean([pred_lgbm, pred_rf, pred_xgb]))

AVG_PREMIUM_EUR = 20.0   # PAD Type A statutory premium

if   ensemble >= 0.35: tier = "HIGH   (above national typical range)"
elif ensemble >= 0.20: tier = "MEDIUM (within national typical range)"
else:                  tier = "LOW    (below national typical range)"

print("\n--- Input ---")
if n_locuinte is not None:
    print(f"  n_locuinte:   {int(n_locuinte):,}")
for feat, val in features_dict.items():
    if feat != "log_n_locuinte":
        print(f"  {feat}: {val}")

print("\n--- Coverage rate prediction ---")
print(f"  LightGBM:     {pred_lgbm:.1%}")
print(f"  RandomForest: {pred_rf:.1%}")
print(f"  XGBoost:      {pred_xgb:.1%}")
print(f"  Ensemble:     {ensemble:.1%}  →  {tier}")

if n_locuinte is not None:
    est_policies    = int(round(n_locuinte * ensemble))
    est_premium_eur = est_policies * AVG_PREMIUM_EUR
    print("\n--- Market sizing (based on ensemble) ---")
    print(f"  Estimated PAD policies:   {est_policies:,}")
    print(f"  Estimated annual premium: {est_premium_eur:,.0f} EUR")
    if current_rate is not None:
        gap_policies = int(round(n_locuinte * max(ensemble - current_rate, 0)))
        gap_eur      = gap_policies * AVG_PREMIUM_EUR
        print(f"\n  Current coverage rate:    {current_rate:.1%}")
        print(f"  Predicted coverage rate:  {ensemble:.1%}")
        print(f"  Opportunity gap:          {gap_policies:,} policies  →  {gap_eur:,.0f} EUR/yr")

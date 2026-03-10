"""
Predict PAD coverage rate for any lat/lon point in Romania.

Finds the nearest locality and runs the ensemble model on its features.
Requires output/model.pkl and output/locality_features.csv (both generated
by running ensembletrees.py).

Usage
-----
  python predict_location.py 45.75 21.23        # Timișoara
  python predict_location.py 44.43 26.10        # Bucharest
  python predict_location.py 46.77 23.59        # Cluj-Napoca
  python predict_location.py 47.16 27.59        # Iași
  python predict_location.py --lat 45.75 --lon 21.23 --model output/model.pkl
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL  = Path(__file__).parent / "output" / "model.pkl"
LOCALITY_TABLE = Path(__file__).parent / "output" / "locality_features.csv"

AVG_PREMIUM_EUR = 20.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance (km) from one point to an array of points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))


# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Predict PAD insurance coverage for any lat/lon location in Romania."
)
parser.add_argument("lat", type=float, help="Latitude  (decimal degrees, e.g. 45.75)")
parser.add_argument("lon", type=float, help="Longitude (decimal degrees, e.g. 21.23)")
parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                    help="Path to model.pkl (default: output/model.pkl)")
parser.add_argument("--top", type=int, default=3,
                    help="Also show the top N nearest localities (default: 3)")
args = parser.parse_args()

for path, label in [(args.model, "output/model.pkl"),
                    (LOCALITY_TABLE, "output/locality_features.csv")]:
    if not path.exists():
        print(f"Error: {label} not found at {path}")
        print("Run  python ensembletrees.py  first to generate model and lookup table.")
        sys.exit(1)

# ── Load ─────────────────────────────────────────────────────────────────────

df = pd.read_csv(LOCALITY_TABLE)
bundle = joblib.load(args.model)
lgbm               = bundle["lgbm"]
rf                 = bundle["rf"]
xgb                = bundle["xgb"]
cv_scores          = bundle["cv_scores"]
feature_cols       = bundle["feature_cols"]
county_mean_lookup = bundle.get("county_mean_lookup", {})

# ── Find nearest locality ─────────────────────────────────────────────────────

lat_col = "loc_lat" if "loc_lat" in df.columns else "lat"
lon_col = "loc_lon" if "loc_lon" in df.columns else "lon"

dist_km = haversine_km(args.lat, args.lon, df[lat_col].values, df[lon_col].values)
df = df.copy()
df["_dist_km"] = dist_km
nearest_idx = int(np.argmin(dist_km))
row = df.iloc[nearest_idx]

# ── Predict ──────────────────────────────────────────────────────────────────

# If county_mean_coverage is missing from the CSV row, fill from the bundle lookup
row_dict = {col: row.get(col, 0) for col in feature_cols}
if row_dict.get("county_mean_coverage", 0) == 0 and "judet" in row:
    import unicodedata
    def _normalize(s):
        s = unicodedata.normalize("NFD", str(s))
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return s.strip().upper()
    key = _normalize(row["judet"])
    looked_up = county_mean_lookup.get(key, 0.0)
    if looked_up:
        row_dict["county_mean_coverage"] = looked_up

X_pred = pd.DataFrame([row_dict])

pred_lgbm = _sigmoid(lgbm.predict(X_pred)[0])
pred_rf   = _sigmoid(rf.predict(X_pred)[0])
pred_xgb  = _sigmoid(xgb.predict(X_pred)[0])

total = sum(cv_scores.values()) if cv_scores else 0
if total > 0:
    w = {k: v / total for k, v in cv_scores.items()}
    ensemble = (pred_lgbm * w["LightGBM"]
                + pred_rf   * w["RandomForest"]
                + pred_xgb  * w["XGBoost"])
else:
    ensemble = float(np.mean([pred_lgbm, pred_rf, pred_xgb]))

# ── Output ───────────────────────────────────────────────────────────────────

if   ensemble >= 0.35: tier = "HIGH   (above national typical range)"
elif ensemble >= 0.20: tier = "MEDIUM (within national typical range)"
else:                  tier = "LOW    (below national typical range)"

n_locuinte    = row.get("n_locuinte", 0)
coverage_rate = row.get("coverage_rate", None)

print(f"\n--- Input ---")
print(f"  Query point: {args.lat:.4f}°N, {args.lon:.4f}°E")

print(f"\n--- Nearest locality ({row['_dist_km']:.1f} km away) ---")
print(f"  {row['localitate']}, {row['judet']}")
if n_locuinte > 0:
    print(f"  Housing units: {int(n_locuinte):,}")
if coverage_rate is not None:
    print(f"  Actual coverage rate: {coverage_rate:.1%}")

print("\n--- Coverage rate prediction ---")
print(f"  LightGBM:     {pred_lgbm:.1%}")
print(f"  RandomForest: {pred_rf:.1%}")
print(f"  XGBoost:      {pred_xgb:.1%}")
print(f"  Ensemble:     {ensemble:.1%}  →  {tier}")

if n_locuinte > 0:
    est_policies = int(round(n_locuinte * ensemble))
    print(f"\n--- Market sizing ---")
    print(f"  Estimated PAD policies:   {est_policies:,}")
    print(f"  Estimated annual premium: {est_policies * AVG_PREMIUM_EUR:,.0f} EUR/yr")
    if coverage_rate is not None:
        gap = int(round(n_locuinte * max(ensemble - coverage_rate, 0)))
        print(f"\n  Actual coverage rate:     {coverage_rate:.1%}")
        print(f"  Predicted coverage rate:  {ensemble:.1%}")
        print(f"  Opportunity gap:          {gap:,} policies  →  {gap * AVG_PREMIUM_EUR:,.0f} EUR/yr")

if args.top > 1:
    top = df.nsmallest(args.top, "_dist_km")[["localitate", "judet", "_dist_km"]]
    print(f"\n--- {args.top} nearest localities ---")
    for _, r in top.iterrows():
        print(f"  {r['_dist_km']:5.1f} km  {r['localitate']}, {r['judet']}")

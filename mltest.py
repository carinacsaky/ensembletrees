"""
MLflow integration for the PAD insurance ensemble model.

Usage
-----
1. Log the existing trained model to MLflow:
       python mltest.py

2. Open the MLflow UI to inspect runs:
       mlflow ui
       # then open http://localhost:5000

3. After logging, load the model from MLflow and predict:
       python mltest.py --predict 45.75 21.23
"""

import argparse
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

BASE_DIR       = Path(__file__).parent
MODEL_PATH     = BASE_DIR / "output" / "model.pkl"
LOCALITY_TABLE = BASE_DIR / "output" / "locality_features.csv"

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))



class PADEnsemble(mlflow.pyfunc.PythonModel):
    """Wraps the three-model ensemble (LightGBM + RF + XGBoost) for MLflow."""

    def load_context(self, context):
        bundle = joblib.load(context.artifacts["model_bundle"])
        self.lgbm         = bundle["lgbm"]
        self.rf           = bundle["rf"]
        self.xgb          = bundle["xgb"]
        self.cv_scores    = bundle["cv_scores"]
        self.feature_cols = bundle["feature_cols"]

    def predict(self, context, model_input):
        X = model_input[self.feature_cols] if hasattr(model_input, "columns") else model_input
        total = sum(self.cv_scores.values()) if self.cv_scores else 0
        if total > 0:
            w = {k: v / total for k, v in self.cv_scores.items()}
        else:
            w = {"LightGBM": 1/3, "RandomForest": 1/3, "XGBoost": 1/3}

        p_lgbm = np.array([_sigmoid(v) for v in self.lgbm.predict(X)])
        p_rf   = np.array([_sigmoid(v) for v in self.rf.predict(X)])
        p_xgb  = np.array([_sigmoid(v) for v in self.xgb.predict(X)])

        ensemble = (p_lgbm * w["LightGBM"]
                    
                    
                    
                    + p_rf   * w["RandomForest"]
                    + p_xgb  * w["XGBoost"])
        return pd.DataFrame({"coverage_rate_pred": ensemble})


# ── Log existing model to MLflow ──────────────────────────────────────────────

def log_model():
    if not MODEL_PATH.exists():
        print(f"Error: {MODEL_PATH} not found. Run python ensembletrees.py first.")
        sys.exit(1)

    bundle = joblib.load(MODEL_PATH)
    cv     = bundle["cv_scores"]

    mlflow.set_experiment("Insurance Coverage")

    with mlflow.start_run(run_name="ensemble_lgbm_rf_xgb"):
        # Log model metrics
        mlflow.log_metric("cv_r2_lightgbm",    cv.get("LightGBM", 0))
        mlflow.log_metric("cv_r2_randomforest", cv.get("RandomForest", 0))
        mlflow.log_metric("cv_r2_xgboost",      cv.get("XGBoost", 0))
        mlflow.log_metric("n_features", len(bundle["feature_cols"]))

        # Log parameters
        mlflow.log_param("models",   "LightGBM + RandomForest + XGBoost")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("target",   "coverage_rate (logit-transformed)")
        mlflow.log_param("country",  "Romania")
        mlflow.log_param("features", ", ".join(bundle["feature_cols"]))

        # Log the model bundle as an artifact + register the pyfunc wrapper
        artifacts = {"model_bundle": str(MODEL_PATH)}
        mlflow.pyfunc.log_model(
            artifact_path="pad_ensemble",
            python_model=PADEnsemble(),
            artifacts=artifacts,
            registered_model_name="PADCoverageEnsemble",
        )

        run_id = mlflow.active_run().info.run_id
        print(f"\nModel logged to MLflow  (run_id: {run_id})")
        print("Open the UI with:  mlflow ui")
        print("Then go to:        http://localhost:5000")
        return run_id


def predict_from_mlflow(lat, lon):
    if not LOCALITY_TABLE.exists():
        print(f"Error: {LOCALITY_TABLE} not found. Run python ensembletrees.py first.")
        sys.exit(1)

    # Load the latest registered model version
    model = mlflow.pyfunc.load_model("models:/PADCoverageEnsemble/latest")

    df = pd.read_csv(LOCALITY_TABLE)
    lat_col = "loc_lat" if "loc_lat" in df.columns else "lat"
    lon_col = "loc_lon" if "loc_lon" in df.columns else "lon"

    dist_km     = haversine_km(lat, lon, df[lat_col].values, df[lon_col].values)
    nearest_idx = int(np.argmin(dist_km))
    row         = df.iloc[nearest_idx]

    # Build feature row (missing cols default to 0)
    bundle       = joblib.load(MODEL_PATH)
    feature_cols = bundle["feature_cols"]
    X_pred       = pd.DataFrame([{col: row.get(col, 0) for col in feature_cols}])

    result   = model.predict(X_pred)
    pred     = float(result["coverage_rate_pred"].iloc[0])
    n_loc    = row.get("n_locuinte", 0)
    act_rate = row.get("coverage_rate", None)
    est      = int(round(n_loc * pred)) if n_loc > 0 else 0
    gap      = int(round(n_loc * max(pred - (act_rate or 0), 0))) if n_loc > 0 else 0

    if   pred >= 0.35: tier = "HIGH"
    elif pred >= 0.20: tier = "MEDIUM"
    else:              tier = "LOW"

    # ── Estimated private insurance premium (insured_value × actuarial rate) ──
    # Base rates: standard Romanian market ranges (‰ of insured value per year)
    avg_flood = row.get("avg_insured_net_flood", 0)
    avg_flood = 0.0 if pd.isna(avg_flood) else float(avg_flood)
    avg_fire  = row.get("avg_insured_net_fire", 0)
    avg_fire  = 0.0 if pd.isna(avg_fire)  else float(avg_fire)
    avg_eq    = avg_flood  # earthquake insured value ≈ flood (same property)

    # ── Location-based risk multipliers ──────────────────────────────────────
    # Flood multiplier: distance to nearest waterway (closer = higher risk)
    dist_ww = float(row.get("dist_to_waterway_km", 2.0) or 2.0)
    if   dist_ww < 0.5: flood_mult = 1.50   # very close to river
    elif dist_ww < 1.5: flood_mult = 1.20   # near river
    elif dist_ww < 5.0: flood_mult = 1.00   # normal
    else:               flood_mult = 0.80   # far from waterways

    # Earthquake multiplier: distance from Vrancea seismic zone (45.7°N, 26.6°E)
    # Vrancea is Romania's main seismic source — closer = higher risk
    vrancea_lat, vrancea_lon = 45.7, 26.6
    dist_vrancea = haversine_km(lat, lon, vrancea_lat, vrancea_lon)
    if   dist_vrancea < 100:  eq_mult = 1.50  # Buzău, Focșani, Brăila area
    elif dist_vrancea < 200:  eq_mult = 1.30  # Bucharest, Bacău area
    elif dist_vrancea < 350:  eq_mult = 1.10  # most of Moldavia/Wallachia
    else:                     eq_mult = 0.80  # western Romania (Timișoara, Cluj)

    # Hazard intensity adjustment (if available from the model features)
    hazard = float(row.get("hazard_intensity", 1.0) or 1.0)
    hazard_mult = max(0.5, min(hazard, 2.0))  # cap between 0.5× and 2×

    prem_flood_lo = avg_flood * 0.0005 * flood_mult * hazard_mult
    prem_flood_hi = avg_flood * 0.0020 * flood_mult * hazard_mult
    prem_eq_lo    = avg_eq   * 0.0003 * eq_mult
    prem_eq_hi    = avg_eq   * 0.0010 * eq_mult
    prem_fire_lo  = avg_fire * 0.0003
    prem_fire_hi  = avg_fire * 0.0008
    prem_total_lo = prem_flood_lo + prem_eq_lo + prem_fire_lo
    prem_total_hi = prem_flood_hi + prem_eq_hi + prem_fire_hi

    # ── Log prediction run to MLflow ──────────────────────────────────────────
    mlflow.set_experiment("Insurance Coverage")
    with mlflow.start_run(run_name=f"predict_{row['localitate']}_{row['judet']}"):
        mlflow.log_param("query_lat",   lat)
        mlflow.log_param("query_lon",   lon)
        mlflow.log_param("locality",    row["localitate"])
        mlflow.log_param("county",      row["judet"])
        mlflow.log_param("tier",        tier)
        mlflow.log_metric("predicted_coverage_rate", round(pred, 4))
        mlflow.log_metric("dist_to_nearest_locality_km", round(float(dist_km[nearest_idx]), 2))
        if act_rate is not None:
            mlflow.log_metric("actual_coverage_rate", round(float(act_rate), 4))
            mlflow.log_metric("gap_pp", round(max(pred - float(act_rate), 0) * 100, 2))
        if n_loc > 0:
            mlflow.log_metric("n_locuinte",          int(n_loc))
            mlflow.log_metric("est_pad_policies",    est)
            mlflow.log_metric("est_premium_eur",     est * 20)
            mlflow.log_metric("opportunity_gap_eur", gap * 20)
        if avg_flood > 0:
            mlflow.log_metric("avg_insured_value_eur",        round(avg_flood))
            mlflow.log_metric("flood_risk_multiplier",        round(flood_mult, 2))
            mlflow.log_metric("earthquake_risk_multiplier",   round(eq_mult, 2))
            mlflow.log_metric("dist_to_waterway_km",          round(dist_ww, 2))
            mlflow.log_metric("dist_to_vrancea_km",           round(dist_vrancea, 1))
            mlflow.log_metric("est_private_premium_low_eur",  round(prem_total_lo))
            mlflow.log_metric("est_private_premium_high_eur", round(prem_total_hi))

    # ── Print summary table ───────────────────────────────────────────────────
    W = 52
    print("\n" + "=" * W)
    print(f"  INSURANCE PREDICTION REPORT")
    print(f"  {row['localitate']}, {row['judet']}  ({dist_km[nearest_idx]:.1f} km from query point)")
    print("=" * W)
    print(f"  {'Coordinates':<30} {lat:.4f}°N, {lon:.4f}°E")
    print(f"  {'PAD coverage (predicted)':<30} {pred:.1%}  [{tier}]")
    if act_rate is not None:
        print(f"  {'PAD coverage (actual)':<30} {act_rate:.1%}")
    if n_loc > 0:
        print(f"  {'Housing units':<30} {int(n_loc):,}")
        print(f"  {'Est. PAD policies':<30} {est:,}")
        print(f"  {'Est. PAD premium/yr':<30} {est * 20:,.0f} EUR")
        print(f"  {'Opportunity gap/yr':<30} {gap * 20:,.0f} EUR")
    if avg_flood > 0:
        print("-" * W)
        print(f"  PRIVATE INSURANCE PREMIUM ESTIMATE")
        print("-" * W)
        print(f"  {'Avg insured value':<30} {avg_flood:,.0f} EUR")
        print(f"  {'Distance to waterway':<30} {dist_ww:.1f} km  (flood ×{flood_mult:.2f})")
        print(f"  {'Distance to Vrancea':<30} {dist_vrancea:.0f} km  (earthquake ×{eq_mult:.2f})")
        print(f"  {'Flood premium/yr':<30} {prem_flood_lo:,.0f} – {prem_flood_hi:,.0f} EUR")
        print(f"  {'Earthquake premium/yr':<30} {prem_eq_lo:,.0f} – {prem_eq_hi:,.0f} EUR")
        print(f"  {'Fire premium/yr':<30} {prem_fire_lo:,.0f} – {prem_fire_hi:,.0f} EUR")
        print(f"  {'TOTAL premium/yr':<30} {prem_total_lo:,.0f} – {prem_total_hi:,.0f} EUR")
    print("=" * W)
    print("  (Results also logged to MLflow: http://localhost:5000)")
    print("=" * W)

    # ── Append to CSV for easy sharing ────────────────────────────────────────
    import csv
    csv_path = BASE_DIR / "output" / "predictions_log.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["lat", "lon", "locality", "county",
                        "predicted_rate", "actual_rate", "tier",
                        "n_locuinte", "est_pad_policies", "opportunity_gap_eur",
                        "avg_insured_value_eur",
                        "premium_low_eur", "premium_high_eur"])
        w.writerow([lat, lon, row["localitate"], row["judet"],
                    round(pred, 4), round(float(act_rate), 4) if act_rate else "",
                    tier, int(n_loc) if n_loc else "",
                    est, gap * 20,
                    round(avg_flood) if avg_flood else "",
                    round(prem_total_lo) if avg_flood else "",
                    round(prem_total_hi) if avg_flood else ""])
    print(f"\n  Saved to output/predictions_log.csv")


# ── Entry point ───────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--predict", nargs=2, type=float, metavar=("LAT", "LON"),
                    help="Predict coverage for a lat/lon using the MLflow model")
args = parser.parse_args()

if args.predict:
    predict_from_mlflow(*args.predict)
else:
    log_model()

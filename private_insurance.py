"""
Private insurance penetration analysis by county and peril.
Standalone script — same ensemble approach as ensembletrees.py.

Computes penetration rate = n_buildings (insurer portfolio) / n_locuinte (total housing stock)
for each of the three perils: FLOOD, EARTHQUAKE, FIRE.

Trains LightGBM + RandomForest + XGBoost per peril.
Target is logit-transformed (same as ensembletrees.py); predictions are sigmoid back-transformed.
CV uses Leave-One-Out because only ~33 counties are available (LOO maximises training data per fold).

Usage
-----
  python private_insurance.py

Requires:
  - data/cache/building_counts_ro.csv  (spatial join cache from ensembletrees.py)
  - data/input/UAT-Grad-cuprindere-...xlsx
  - DB connection (for NUTS3 county attributes) defined in .env
"""

import os
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

XLSX_PATH             = BASE_DIR / "data" / "input" / "UAT-Grad-cuprindere-in-asigurare-la-31-01-2026.xlsx"
BUILDING_COUNTS_CACHE = BASE_DIR / "data" / "cache" / "building_counts_ro.csv"
COUNTRY_CODE          = "RO"
OUTPUT_DIR            = BASE_DIR / "output"
PRIVATE_MODEL_PATH    = OUTPUT_DIR / "private_model.pkl"

DB_CONFIG = dict(
    host     = os.getenv("DB_HOST", "10.100.0.10"),
    dbname   = os.getenv("DB_NAME", "riskdata"),
    user     = os.getenv("DB_USER", "geoserver"),
    password = os.getenv("DB_PASSWORD", ""),
    port     = int(os.getenv("DB_PORT", "31183")),
)

PERILS = ["flood", "earthquake", "fire"]

FEATURE_COLS = [
    "urbn_type",
    "mount_type",
    "coast_type",
    "lat",
    "lon",
    "log_n_locuinte_county",
    "pop_density",
    "avg_waterway_dist_km",
]

WATERWAY_DIST_CACHE = BASE_DIR / "data" / "cache" / "locality_waterway_dist_ro.csv"


def _normalize(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.strip().upper()


def _make_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}",
        connect_args={
            "keepalives": 1,
            "keepalives_idle": 5,
            "keepalives_interval": 3,
            "keepalives_count": 10,
            "connect_timeout": 120,
        }
    )


def load_xlsx_data():
    df = pd.read_excel(XLSX_PATH, header=1)
    df = df.iloc[2:].copy()
    df.columns = [
        "judet", "tip", "localitate",
        "n_locuinte", "n_pad", "coverage_rate",
        "n_locuinte_public", "n_pad_public", "coverage_rate_public",
    ]
    df["judet"] = df["judet"].ffill()
    for col in ["n_locuinte", "n_pad"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["judet", "tip", "n_locuinte"])
    df = df[df["n_locuinte"] > 0].copy()
    df["judet_key"] = df["judet"].apply(_normalize)
    print(f"  Localities: {len(df):,}  |  Counties: {df['judet'].nunique()}")
    return df


def load_nuts3_features():
    engine = _make_engine()
    query = text("""
        SELECT
            name_latn,
            urbn_type,
            mount_type,
            coast_type,
            ST_Y(ST_Centroid(ST_Transform(geom, 4326))) AS lat,
            ST_X(ST_Centroid(ST_Transform(geom, 4326))) AS lon,
            ST_Area(ST_Transform(geom, 3857)) / 1e6 AS area_km2
        FROM nuts3_eu_admin
        WHERE cntr_code = :country
    """)
    df = pd.read_sql(query, engine, params={"country": COUNTRY_CODE})
    for col in ["urbn_type", "mount_type", "coast_type", "lat", "lon", "area_km2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["judet_key"] = df["name_latn"].apply(_normalize)
    print(f"  NUTS3 counties loaded: {len(df)}")
    return df


def load_waterway_dist():
    df = pd.read_csv(WATERWAY_DIST_CACHE)
    df["judet_key"] = df["judet"].apply(_normalize)
    county_dist = (
        df.groupby("judet_key")["dist_to_waterway_km"]
        .mean()
        .reset_index()
        .rename(columns={"dist_to_waterway_km": "avg_waterway_dist_km"})
    )
    print(f"  Waterway distances loaded ({len(county_dist)} counties)")
    return county_dist


def load_building_counts():
    if BUILDING_COUNTS_CACHE.exists():
        df = pd.read_csv(BUILDING_COUNTS_CACHE)
        print(f"  Loaded building counts from cache ({len(df)} counties)")
        return df
    print("  No cache found — run ensembletrees.py first to generate building_counts_ro.csv")
    raise FileNotFoundError(BUILDING_COUNTS_CACHE)


def build_county_dataset():
    print("Loading PAD xlsx data...")
    df_xlsx = load_xlsx_data()

    county_housing = (
        df_xlsx.groupby("judet_key")["n_locuinte"]
        .sum()
        .reset_index()
        .rename(columns={"n_locuinte": "n_locuinte_county"})
    )
    county_housing["log_n_locuinte_county"] = np.log1p(county_housing["n_locuinte_county"])

    print("\nLoading building counts from cache...")
    df_buildings = load_building_counts()

    print("\nLoading NUTS3 county attributes from DB...")
    df_nuts3 = load_nuts3_features()

    print("\nLoading waterway distances from cache...")
    df_waterway = load_waterway_dist()

    df = county_housing.merge(df_buildings, on="judet_key", how="inner")
    df = df.merge(
        df_nuts3[["judet_key", "urbn_type", "mount_type", "coast_type", "lat", "lon", "area_km2"]],
        on="judet_key", how="inner"
    )
    df = df.merge(df_waterway, on="judet_key", how="left")

    df["pop_density"] = df["n_locuinte_county"] / df["area_km2"].replace(0, np.nan)

    print(f"\n  Counties after join: {len(df)}")

    for peril in PERILS:
        df[f"private_penetration_{peril}"] = (
            df[f"n_buildings_{peril}"] / df["n_locuinte_county"]
        ).clip(0, 1)

    return df


def train_models(X, y_orig, county_names, peril):
    """Train LightGBM + RandomForest + XGBoost on logit-transformed target.
    Uses Leave-One-Out CV (n=33 is too small for k-fold). Returns fitted models and CV R² scores."""
    eps = 1e-6
    y = np.log((y_orig + eps) / (1 - y_orig + eps))  # logit transform

    loo = LeaveOneOut()

    lgbm = LGBMRegressor(
        n_estimators=100, learning_rate=0.05, num_leaves=4,
        min_child_samples=5, random_state=42, verbose=-1,
    )
    rf = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=5, max_depth=3, random_state=42, n_jobs=-1,
    )
    xgb = XGBRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=2,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, tree_method="hist",
    )

    cv_preds  = {}
    cv_scores = {}

    print(f"\n--- {peril.upper()} ---")
    for name, model in [("LightGBM", lgbm), ("RandomForest", rf), ("XGBoost", xgb)]:
        y_pred_logit = cross_val_predict(model, X, y, cv=loo)
        y_pred = 1 / (1 + np.exp(-np.clip(y_pred_logit, -6, 6)))
        r2  = r2_score(y_orig, y_pred)
        mae = mean_absolute_error(y_orig, y_pred)
        cv_preds[name]  = y_pred
        cv_scores[name] = max(r2, 0.0)
        print(f"  {name:<15} R²={r2:.3f}  MAE={mae*100:.2f}pp")

    # Ensemble: weighted by R² if any model beats the mean, else equal weights
    total = sum(cv_scores.values())
    if total > 0:
        w = {k: v / total for k, v in cv_scores.items()}
    else:
        w = {k: 1.0 / len(cv_scores) for k in cv_scores}
    ensemble = (
        cv_preds["LightGBM"]    * w["LightGBM"] +
        cv_preds["RandomForest"] * w["RandomForest"] +
        cv_preds["XGBoost"]     * w["XGBoost"]
    )
    ens_r2  = r2_score(y_orig, ensemble)
    ens_mae = mean_absolute_error(y_orig, ensemble)
    print(f"  {'Ensemble':<15} R²={ens_r2:.3f}  MAE={ens_mae*100:.2f}pp")

    # Feature importance (average across models)
    lgbm.fit(X, y)
    rf.fit(X, y)
    xgb.fit(X, y)
    imp = (lgbm.feature_importances_ + rf.feature_importances_ + xgb.feature_importances_) / 3
    imp_series = pd.Series(imp, index=X.columns).sort_values(ascending=False)
    print("  Top features:")
    for feat, score in imp_series.head(5).items():
        print(f"    {feat:<35} {score:.1f}")

    results = pd.DataFrame({
        "county":    county_names,
        "actual":    y_orig,
        "predicted": ensemble,
    }).sort_values("actual", ascending=False)

    print(f"\n  {'County':<25} {'Actual':>8} {'Predicted':>10}")
    print("  " + "-" * 45)
    for _, row in results.iterrows():
        print(f"  {row['county']:<25} {row['actual']:8.2%} {row['predicted']:10.2%}")

    return lgbm, rf, xgb, cv_scores


def main():
    df = build_county_dataset()

    print("\n--- Private penetration rate summary ---")
    for peril in PERILS:
        col = f"private_penetration_{peril}"
        s = df[col]
        print(f"  {peril:12s}  mean={s.mean():.2%}  min={s.min():.2%}  max={s.max():.2%}")

    feats = [c for c in FEATURE_COLS if c in df.columns]

    bundle_models = {}
    for peril in PERILS:
        # Exclude counties with zero penetration — the insurer has no presence there
        # at all (different problem from low penetration; their logit is -inf)
        mask = df[f"private_penetration_{peril}"] > 0
        df_peril = df[mask]
        X = df_peril[feats].fillna(0)
        county_names = df_peril["county_name"].values
        y = df_peril[f"private_penetration_{peril}"].values
        print(f"\n  [{peril}] Training on {len(df_peril)} counties with non-zero penetration "
              f"(excluded {(~mask).sum()} zero-penetration counties)")
        lgbm, rf, xgb, cv_scores = train_models(X, y, county_names, peril)
        bundle_models[peril] = {
            "lgbm": lgbm, "rf": rf, "xgb": xgb, "cv_scores": cv_scores,
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {"models": bundle_models, "feature_cols": feats}
    joblib.dump(bundle, PRIVATE_MODEL_PATH)
    print(f"\n  Model bundle saved to {PRIVATE_MODEL_PATH}")

    # --- Save private penetration table to CSV ---
    priv_cols = ["county_name", "n_locuinte_county"] + \
                [f"private_penetration_{p}" for p in PERILS]
    priv_csv = OUTPUT_DIR / "private_penetration.csv"
    df[priv_cols].sort_values("private_penetration_flood", ascending=False).to_csv(
        priv_csv, index=False, float_format="%.6f"
    )
    print(f"  Private penetration table saved to {priv_csv}")

    # --- Cross-analysis: PAD opportunity vs private penetration ---
    pad_csv = OUTPUT_DIR / "county_summary.csv"
    if not pad_csv.exists():
        print("\n  (Skipping cross-analysis — run ensembletrees.py first to generate county_summary.csv)")
        return

    pad = pd.read_csv(pad_csv)
    pad["judet_key"] = pad["judet"].apply(_normalize)

    priv = df[priv_cols].copy()
    priv["judet_key"] = priv["county_name"].apply(_normalize)
    priv["private_penetration_avg"] = priv[
        [f"private_penetration_{p}" for p in PERILS]
    ].mean(axis=1)

    cross = pad.merge(priv[["judet_key", "private_penetration_avg"]], on="judet_key", how="inner")
    cross["private_pct"] = cross["private_penetration_avg"] * 100
    # combined_score: PAD locality-level opportunity weighted by how under-penetrated private is
    cross["combined_score"] = cross["opportunity_eur"] * (1 - cross["private_penetration_avg"])

    cross_out = cross[["judet", "n_locuinte", "actual_coverage", "predicted_coverage",
                        "opportunity_eur", "private_pct", "combined_score"]]
    cross_out = cross_out.sort_values("combined_score", ascending=False)
    cross_out.to_csv(OUTPUT_DIR / "cross_analysis.csv", index=False, float_format="%.4f")

    print("\n--- Cross-analysis: PAD opportunity × private under-penetration ---")
    print(f"  {'County':<20} {'PAD actual':>10} {'PAD pred':>9} {'PAD opp EUR':>12} {'Private':>8} {'Score':>12}")
    print("  " + "-" * 77)
    for _, r in cross_out.head(15).iterrows():
        print(f"  {r['judet']:<20} {r['actual_coverage']:10.1%} {r['predicted_coverage']:9.1%} "
              f"{r['opportunity_eur']:12,.0f} {r['private_pct']:7.1f}% {r['combined_score']:12,.0f}")
    print(f"\n  Saved to {OUTPUT_DIR / 'cross_analysis.csv'}")


if __name__ == "__main__":
    main()

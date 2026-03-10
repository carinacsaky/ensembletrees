"""
Italy locality-level insurance coverage analysis.
Adapted from ensembletrees.py (Romania PAD model).

Pipeline is identical — only the data loading layer changes.

Data you need
─────────────
1. Insurance coverage per comune  ← the one thing this script can't run without
   Any CSV / Excel with at minimum:
     - province name  (e.g. "Milano", "Roma")
     - comune name    (e.g. "Cologno Monzese")
     - n_abitazioni   total housing units
     - n_insured      insured dwellings (or a coverage_rate directly)
   Possible sources:
     - ANIA (Associazione Nazionale fra le Imprese Assicuratrici) — market stats
     - A specific insurer's portfolio export
     - IVASS (insurance regulator) open data: https://www.ivass.it/open-data

2. Waterways GeoJSON  ← download once from HOT OSM export
   https://export.hotosm.org  → Italy → Waterways → GeoJSON
   Save to: data/input/hotosm_ita_waterways_lines.geojson

3. Unemployment per comune  ← ISTAT
   https://esploradati.istat.it  → search "disoccupati comuni"
   Or: ISTAT Censimento 2021, variable "tasso disoccupazione"
   Save anywhere and point UNEMPLOYMENT_PATH at it.

Everything else (NUTS3, building footprints, geocoding, ML pipeline)
is already available and requires no changes.

Usage
─────
  python italy_insurance.py
"""

import os
import re
import time
import copy
import joblib
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shap
import requests
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# ─── DB (same as Romania — just change COUNTRY_CODE) ─────────────────────────
DB_CONFIG = dict(
    host     = os.getenv("DB_HOST", "10.100.0.10"),
    dbname   = os.getenv("DB_NAME", "riskdata"),
    user     = os.getenv("DB_USER", "geoserver"),
    password = os.getenv("DB_PASSWORD", ""),
    port     = int(os.getenv("DB_PORT", "31183")),
)
COUNTRY_CODE = "IT"

# ─── Paths ────────────────────────────────────────────────────────────────────
# TODO: point this at your insurance coverage file once you have it
INSURANCE_DATA_PATH = BASE_DIR / "data" / "input" / "italy_insurance_coverage.csv"

WATERWAY_PATH  = BASE_DIR / "data" / "input"  / "hotosm_ita_waterways_lines.geojson"
WATERWAY_CACHE = BASE_DIR / "data" / "cache"  / "comune_waterway_dist_it.csv"
WATERWAY_TYPES = ("river", "canal", "stream")

BUILDING_COUNTS_CACHE  = BASE_DIR / "data" / "cache" / "building_counts_it.csv"
COORDS_CACHE           = BASE_DIR / "data" / "cache" / "comune_coords_it.csv"
UNEMPLOYMENT_CACHE     = BASE_DIR / "data" / "cache" / "unemployment_it.csv"

# TODO: point at your ISTAT unemployment file once you have it
UNEMPLOYMENT_PATH = BASE_DIR / "data" / "input" / "istat_disoccupati_comuni.csv"

(BASE_DIR / "data" / "cache").mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = BASE_DIR / "output_italy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "model_italy.pkl"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ─── Insurance product constants ─────────────────────────────────────────────
# TODO: update these for whichever product you are modelling
AVG_PREMIUM_EUR        = 150.0    # rough average Italian home insurance premium
SUM_INSURED_EUR        = 150_000  # rough average Italian dwelling value
VULNERABILITY_FACTOR   = 0.25
MANDATORY_SCENARIO_RATE = 0.70
CAT_STRESS_MULTIPLIER  = 3.0
BOOTSTRAP_ITERATIONS   = 100

TARGET_COL   = "coverage_rate"
FOCUS_PROVINCE = "Milano"   # drill-down province in output plots


# ─── Column mapping for your insurance file ───────────────────────────────────
# Edit these to match whatever columns your file actually has.
COL_MAP = {
    "provincia":      "provincia",    # province / NUTS3 name
    "comune":         "comune",       # municipality name
    "n_abitazioni":   "n_abitazioni", # total housing units
    "n_insured":      "n_insured",    # insured dwellings  (OR provide coverage_rate directly)
    "coverage_rate":  None,           # set to a column name if pre-computed; else derived
}


# ═══════════════════════════════════════════════════════════════════════════════
# 0. HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize(s):
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.strip().upper()

def _make_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}",
        connect_args={"keepalives": 1, "keepalives_idle": 5,
                      "keepalives_interval": 3, "keepalives_count": 10,
                      "connect_timeout": 120},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_insurance_data(path=INSURANCE_DATA_PATH, col_map=COL_MAP):
    """
    Load locality-level insurance coverage.

    Expects a CSV or Excel file.  Edit COL_MAP at the top of this file to match
    your column names.  The output DataFrame always uses the standard internal
    names: provincia, comune, n_abitazioni, n_insured, coverage_rate.

    Settlement type (tip_code) is derived from housing stock size since Italy
    does not have a direct equivalent to Romania's Municipiu/Oras/Comuna:
      1 = large  (n_abitazioni > 20,000)  — city
      2 = medium (n_abitazioni 5,000–20,000) — town
      3 = small  (n_abitazioni < 5,000)   — commune
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Insurance data not found: {path}\n"
            "Download or export a comune-level coverage file and update INSURANCE_DATA_PATH."
        )

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Rename columns to internal standard names
    rename = {v: k for k, v in col_map.items() if v and v in df.columns}
    df = df.rename(columns=rename)

    for col in ["n_abitazioni", "n_insured"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["provincia", "comune", "n_abitazioni"])
    df = df[df["n_abitazioni"] > 0].copy()

    # Derive coverage_rate if not supplied directly
    if "coverage_rate" not in df.columns or df["coverage_rate"].isna().all():
        if "n_insured" not in df.columns:
            raise ValueError(
                "Need either a 'coverage_rate' column or an 'n_insured' column."
            )
        df["coverage_rate"] = (df["n_insured"] / df["n_abitazioni"]).clip(0, 1)
    df["coverage_rate"] = pd.to_numeric(df["coverage_rate"], errors="coerce").clip(0, 1)

    # Derive n_insured if only coverage_rate was provided
    if "n_insured" not in df.columns:
        df["n_insured"] = (df["coverage_rate"] * df["n_abitazioni"]).round().astype(int)

    # Settlement type proxy from housing stock size
    df["tip_code"] = np.select(
        [df["n_abitazioni"] > 20_000, df["n_abitazioni"] > 5_000],
        [1, 2],
        default=3,
    )

    df["provincia_key"] = df["provincia"].apply(_normalize)
    df["comune_key"]    = df["comune"].apply(_normalize)

    print(f"  Comuni: {len(df):,}  |  Provinces: {df['provincia'].nunique()}")
    return df


def load_nuts3_features(country_code=COUNTRY_CODE):
    """Query NUTS3 province attributes from DB — identical to Romania version."""
    engine = _make_engine()
    query = text("""
        SELECT
            name_latn,
            urbn_type, mount_type, coast_type,
            ST_Y(ST_Centroid(ST_Transform(geom, 4326))) AS lat,
            ST_X(ST_Centroid(ST_Transform(geom, 4326))) AS lon
        FROM nuts3_eu_admin
        WHERE cntr_code = :country
    """)
    df = pd.read_sql(query, engine, params={"country": country_code})
    for col in ["urbn_type", "mount_type", "coast_type", "lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["provincia_key"] = df["name_latn"].apply(_normalize)
    print(f"  NUTS3 provinces loaded: {len(df)}")
    return df


def load_building_counts(country_code=COUNTRY_CODE, cache_path=BUILDING_COUNTS_CACHE):
    """Building footprint stats per province — identical query to Romania."""
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"  Loaded building counts from cache ({len(df)} provinces)")
        return df

    print("  No cache — running spatial join (several minutes) ...")
    engine = _make_engine()
    query = text("""
        SELECT
            n.name_latn AS province_name,
            b.covered_peril,
            COUNT(b.id)                                        AS n_buildings,
            SUM(b.insured_value_net)                           AS sum_insured_net,
            SUM(b.insured_value_gross)                         AS sum_insured_gross,
            AVG(b.insured_value_net)                           AS avg_insured_net,
            AVG(ST_Area(ST_Transform(b.geom, 3857)))           AS avg_footprint_m2
        FROM nuts3_eu_admin n
        LEFT JOIN building_footprints_partition b
            ON ST_Intersects(b.geom, n.geom)
           AND b.covered_peril IN ('FLOOD', 'EARTHQUAKE', 'FIRE')
        WHERE n.cntr_code = :country
        GROUP BY n.name_latn, b.covered_peril
    """)
    df = pd.read_sql(query, engine, params={"country": country_code})

    for col in ["n_buildings", "sum_insured_net", "sum_insured_gross",
                "avg_insured_net", "avg_footprint_m2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    pivot_count = df.pivot_table(index="province_name", columns="covered_peril",
                                 values="n_buildings", aggfunc="sum", fill_value=0)
    pivot_count.columns = [f"n_buildings_{c.lower()}" for c in pivot_count.columns]

    pivot_net = df.pivot_table(index="province_name", columns="covered_peril",
                               values="sum_insured_net", aggfunc="sum", fill_value=0)
    pivot_net.columns = [f"sum_insured_net_{c.lower()}" for c in pivot_net.columns]

    pivot_avg = df.pivot_table(index="province_name", columns="covered_peril",
                               values="avg_insured_net", aggfunc="mean", fill_value=0)
    pivot_avg.columns = [f"avg_insured_net_{c.lower()}" for c in pivot_avg.columns]

    pivot_gross = df.pivot_table(index="province_name", columns="covered_peril",
                                 values="sum_insured_gross", aggfunc="sum", fill_value=0)
    pivot_gross.columns = [f"sum_insured_gross_{c.lower()}" for c in pivot_gross.columns]

    pivot_area = df.pivot_table(index="province_name", columns="covered_peril",
                                values="avg_footprint_m2", aggfunc="mean", fill_value=0)
    pivot_area.columns = [f"avg_footprint_m2_{c.lower()}" for c in pivot_area.columns]

    pivot = (pivot_count.join(pivot_net).join(pivot_avg)
                        .join(pivot_gross).join(pivot_area).reset_index())

    n_cols = [c for c in pivot.columns if c.startswith("n_buildings_")]
    pivot["total_buildings"]     = pivot[n_cols].sum(axis=1)
    pivot["log_total_buildings"] = np.log1p(pivot["total_buildings"])

    for peril in ["flood", "earthquake", "fire"]:
        net_col, gross_col = f"sum_insured_net_{peril}", f"sum_insured_gross_{peril}"
        if net_col in pivot.columns and gross_col in pivot.columns:
            pivot[f"retention_ratio_{peril}"] = np.where(
                pivot[gross_col] > 0,
                (pivot[net_col] / pivot[gross_col]).clip(0, 1), np.nan,
            )

    area_cols = [c for c in pivot.columns if c.startswith("avg_footprint_m2_")]
    if area_cols:
        pivot["avg_footprint_m2"]     = pivot[area_cols].replace(0, np.nan).mean(axis=1)
        pivot["log_avg_footprint_m2"] = np.log1p(pivot["avg_footprint_m2"].fillna(0))

    pivot["provincia_key"] = pivot["province_name"].apply(_normalize)
    pivot.to_csv(cache_path, index=False)
    print(f"  Building counts for {len(pivot)} provinces — saved to {cache_path}")
    return pivot


def load_comune_coordinates(df_ins, cache_path=COORDS_CACHE):
    """Geocode comuni via Google Maps API, cached to CSV."""
    if cache_path.exists():
        df_coords = pd.read_csv(cache_path)
        n_ok = df_coords["loc_lat"].notna().sum()
        print(f"  Loaded coordinates from cache ({n_ok}/{len(df_coords)} geocoded)")
        return df_coords

    localities = df_ins[["provincia", "comune"]].drop_duplicates().reset_index(drop=True)
    print(f"  Geocoding {len(localities):,} comuni via Google Maps ...")
    rows = []
    for i, row in localities.iterrows():
        address = f"{row['comune']}, {row['provincia']}, Italy"
        loc_lat = loc_lon = None
        try:
            resp = requests.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": address, "key": GOOGLE_API_KEY, "region": "it"},
                timeout=10,
            )
            data = resp.json()
            if data.get("status") == "OK" and data.get("results"):
                geo = data["results"][0]["geometry"]["location"]
                loc_lat, loc_lon = geo["lat"], geo["lng"]
        except Exception:
            pass
        rows.append({"provincia": row["provincia"], "comune": row["comune"],
                     "loc_lat": loc_lat, "loc_lon": loc_lon})
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(localities)} done ...")
        time.sleep(0.05)

    df_coords = pd.DataFrame(rows)
    df_coords.to_csv(cache_path, index=False)
    print(f"  Geocoded {df_coords['loc_lat'].notna().sum()}/{len(df_coords)} comuni")
    return df_coords


def load_waterway_features(df_coords, waterway_path=WATERWAY_PATH,
                           cache_path=WATERWAY_CACHE, waterway_types=WATERWAY_TYPES):
    """Distance to nearest waterway per comune — identical logic to Romania."""
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
        df_coords = df_coords.merge(
            cache[["provincia", "comune", "dist_to_waterway_km"]],
            on=["provincia", "comune"], how="left",
        )
        print(f"  Loaded waterway distances from cache "
              f"({df_coords['dist_to_waterway_km'].notna().sum()}/{len(df_coords)} comuni)")
        return df_coords

    if not waterway_path.exists():
        print(f"  Waterway file not found: {waterway_path} — skipping")
        return df_coords

    waterways = gpd.read_file(waterway_path)
    waterways = waterways[waterways["waterway"].isin(waterway_types)][["geometry"]].to_crs("EPSG:32632")  # UTM zone 32N — Italy

    df_valid = df_coords.dropna(subset=["loc_lat", "loc_lon"]).reset_index(drop=True)
    gdf_loc = gpd.GeoDataFrame(
        df_valid[["provincia", "comune"]].copy(),
        geometry=gpd.points_from_xy(df_valid["loc_lon"], df_valid["loc_lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32632")

    print(f"  Computing nearest waterway for {len(gdf_loc):,} comuni ...")
    nearest = gpd.sjoin_nearest(gdf_loc, waterways, how="left", distance_col="dist_m")
    nearest = nearest.groupby(nearest.index)["dist_m"].min()
    gdf_loc["dist_to_waterway_km"] = (gdf_loc.index.map(nearest) / 1000).round(3)

    df_coords = df_coords.merge(
        gdf_loc[["provincia", "comune", "dist_to_waterway_km"]],
        on=["provincia", "comune"], how="left",
    )
    df_coords[["provincia", "comune", "dist_to_waterway_km"]].to_csv(cache_path, index=False)
    print(f"  Waterway distances computed for "
          f"{df_coords['dist_to_waterway_km'].notna().sum()} comuni")
    return df_coords


def load_unemployment_data(df_ins, path=UNEMPLOYMENT_PATH, cache_path=UNEMPLOYMENT_CACHE):
    """
    Load ISTAT unemployment data per comune.

    TODO: once you have the file, update the column mapping below.
    Expected columns (any CSV/Excel from ISTAT):
      - provincia  — province name
      - comune     — municipality name
      - disoccupati or tasso_disoccupazione — unemployed count or rate

    If you have a rate directly, set disoccupati=None and provide tasso_disoccupazione.
    We compute: someri_rate = disoccupati / n_abitazioni  (if counts)
                           or tasso_disoccupazione / 100   (if rate)
    """
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
        print(f"  Loaded unemployment from cache ({cache['disoccupati'].notna().sum()} comuni)")
        return df_ins.merge(cache[["provincia", "comune", "disoccupati"]],
                            on=["provincia", "comune"], how="left")

    if not path.exists():
        print(f"  Unemployment file not found: {path} — skipping (feature will be absent)")
        return df_ins

    # ── TODO: adapt this block to your actual ISTAT file columns ─────────────
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
    # Example column rename — change to match your file:
    # df = df.rename(columns={"NOME_COMUNE": "comune", "NOME_PROVINCIA": "provincia",
    #                          "DISOCCUPATI": "disoccupati"})
    df["disoccupati"] = pd.to_numeric(df.get("disoccupati"), errors="coerce")
    df["provincia_key"] = df["provincia"].apply(_normalize)
    df["comune_key"]    = df["comune"].apply(_normalize)

    out = df_ins.merge(
        df[["provincia_key", "comune_key", "disoccupati"]],
        left_on=["provincia_key", "comune_key"],
        right_on=["provincia_key", "comune_key"],
        how="left",
    )
    out[["provincia", "comune", "disoccupati"]].to_csv(cache_path, index=False)
    print(f"  Unemployment matched: {out['disoccupati'].notna().sum()}/{len(out)} comuni")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "tip_code",               # 1=large city, 2=town, 3=small comune
    "log_n_abitazioni",       # log housing stock
    "urbn_type",              # from NUTS3
    "mount_type",
    "coast_type",
    "loc_lat",
    "loc_lon",
    "log_total_buildings",
    "n_buildings_flood",
    "n_buildings_fire",
    "n_buildings_earthquake",
    "n_active_perils",
    "avg_insured_net_flood",
    "avg_insured_net_fire",
    "retention_ratio_flood",
    "retention_ratio_earthquake",
    "retention_ratio_fire",
    "log_avg_footprint_m2",
    "hazard_intensity",
    "dist_to_waterway_km",
    "someri_rate",            # unemployment proxy (if ISTAT data available)
    "county_mean_coverage",   # LOO province mean coverage (strong fixed effect)
]


def build_features(df_ins, df_nuts3, df_buildings, df_coords=None):
    df = df_ins.copy()
    df["log_n_abitazioni"] = np.log1p(df["n_abitazioni"])

    # Province-level NUTS3 attributes
    nuts3_cols = ["provincia_key", "urbn_type", "mount_type", "coast_type", "lat", "lon"]
    df = df.merge(df_nuts3[nuts3_cols], on="provincia_key", how="left")

    # Building counts (province-level)
    bld_base = ["provincia_key", "log_total_buildings",
                "n_buildings_flood", "n_buildings_fire", "n_buildings_earthquake"]
    bld_extra = [c for c in df_buildings.columns
                 if c in ("avg_insured_net_flood", "avg_insured_net_fire",
                          "retention_ratio_flood", "retention_ratio_earthquake",
                          "retention_ratio_fire", "log_avg_footprint_m2")]
    bld_available = [c for c in bld_base if c in df_buildings.columns]
    if bld_available and len(df_buildings) > 0:
        df = df.merge(df_buildings[bld_available + bld_extra], on="provincia_key", how="left")
    else:
        for col in bld_base[1:] + bld_extra:
            df[col] = 0.0

    # Locality coordinates + waterway distance
    if df_coords is not None:
        coord_cols = ["provincia", "comune", "loc_lat", "loc_lon"]
        if "dist_to_waterway_km" in df_coords.columns:
            coord_cols.append("dist_to_waterway_km")
        df = df.merge(df_coords[coord_cols], on=["provincia", "comune"], how="left")
        df["loc_lat"] = df["loc_lat"].fillna(df["lat"])
        df["loc_lon"] = df["loc_lon"].fillna(df["lon"])
        if "dist_to_waterway_km" in df.columns:
            df["dist_to_waterway_km"] = df["dist_to_waterway_km"].fillna(
                df["dist_to_waterway_km"].median()
            )
    else:
        df["loc_lat"] = df["lat"]
        df["loc_lon"] = df["lon"]

    # Derived features
    df["n_active_perils"] = (
        (df["n_buildings_flood"].fillna(0)      > 0).astype(int) +
        (df["n_buildings_earthquake"].fillna(0) > 0).astype(int) +
        (df["n_buildings_fire"].fillna(0)       > 0).astype(int)
    )
    df["hazard_intensity"] = (
        df["n_buildings_flood"].fillna(0)      * 0.4 +
        df["n_buildings_earthquake"].fillna(0) * 0.4 +
        df["n_buildings_fire"].fillna(0)       * 0.2
    )
    max_hi = df["hazard_intensity"].max()
    if max_hi > 0:
        df["hazard_intensity"] /= max_hi

    # Unemployment rate
    if "disoccupati" in df.columns:
        df["someri_rate"] = (df["disoccupati"] / df["n_abitazioni"].replace(0, np.nan)).clip(0)
        prov_median = df.groupby("provincia_key")["someri_rate"].transform("median")
        df["someri_rate"] = df["someri_rate"].fillna(prov_median).fillna(df["someri_rate"].median())

    # Province-mean coverage (LOO)
    county_sum   = df.groupby("provincia_key")["coverage_rate"].transform("sum")
    county_count = df.groupby("provincia_key")["coverage_rate"].transform("count")
    df["county_mean_coverage"] = (county_sum - df["coverage_rate"]) / (county_count - 1)
    df["county_mean_coverage"] = df["county_mean_coverage"].fillna(df["coverage_rate"])

    n_unmatched = df["urbn_type"].isna().sum()
    if n_unmatched:
        print(f"  Warning: {n_unmatched} comuni unmatched to NUTS3")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODEL PREP  (identical to Romania)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_model_data(df):
    df = df[df[TARGET_COL].notna()].copy().reset_index(drop=True)
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].copy()

    _MEDIAN_FILL = {
        "avg_insured_net_flood", "avg_insured_net_fire",
        "retention_ratio_flood", "retention_ratio_earthquake", "retention_ratio_fire",
        "log_avg_footprint_m2", "dist_to_waterway_km", "someri_rate",
    }
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median() if col in _MEDIAN_FILL else 0)

    y_orig  = df[TARGET_COL].values
    eps     = 1e-6
    y_safe  = np.clip(y_orig, eps, 1 - eps)
    y       = np.log(y_safe / (1 - y_safe))
    return X, y, y_orig, df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING  (identical to Romania)
# ═══════════════════════════════════════════════════════════════════════════════

def train_models(X, y, y_orig):
    y_bins    = pd.qcut(y_orig, q=5, labels=False, duplicates="drop")
    kf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(kf.split(X, y_bins))

    lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         min_child_samples=20, random_state=42, verbose=-1)
    rf   = RandomForestRegressor(n_estimators=300, min_samples_leaf=10,
                                 random_state=42, n_jobs=-1)
    xgb  = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, verbosity=0, tree_method="hist")

    cv_preds  = {}
    cv_scores = {}
    for name, model in [("LightGBM", lgbm), ("RandomForest", rf), ("XGBoost", xgb)]:
        y_pred_logit = cross_val_predict(model, X, y, cv=cv_splits)
        y_pred = np.clip(1 / (1 + np.exp(-y_pred_logit)), 0, 1)
        r2   = r2_score(y_orig, y_pred)
        mae  = mean_absolute_error(y_orig, y_pred)
        rmse = np.sqrt(mean_squared_error(y_orig, y_pred))
        cv_preds[name]  = y_pred
        cv_scores[name] = max(r2, 0.0)
        print(f"{name}\n  R²: {r2:.3f}  MAE: {mae*100:.2f}pp  RMSE: {rmse*100:.2f}pp\n")

    lgbm.fit(X, y)
    rf.fit(X, y)
    xgb.fit(X, y)
    return lgbm, rf, xgb, cv_preds, cv_scores


def save_models(lgbm, rf, xgb, cv_scores, feature_cols, county_mean_lookup, path=MODEL_PATH):
    bundle = {"lgbm": lgbm, "rf": rf, "xgb": xgb, "cv_scores": cv_scores,
              "feature_cols": feature_cols, "county_mean_lookup": county_mean_lookup}
    joblib.dump(bundle, path)
    print(f"  Model saved to {path}")


def predict_from_features(features_dict, model_path=MODEL_PATH):
    """
    Predict coverage rate for a single location.

    Pass 'provincia' to auto-fill county_mean_coverage from the bundle.

    Example:
        predict_from_features({
            "provincia":     "Milano",
            "tip_code":      1,
            "log_n_abitazioni": np.log1p(15_000),
            "loc_lat":       45.46,
            "loc_lon":       9.19,
            "dist_to_waterway_km": 1.5,
        })
    """
    bundle             = joblib.load(model_path)
    lgbm               = bundle["lgbm"]
    rf                 = bundle["rf"]
    xgb                = bundle["xgb"]
    cv_scores          = bundle["cv_scores"]
    feature_cols       = bundle["feature_cols"]
    county_mean_lookup = bundle.get("county_mean_lookup", {})

    features_dict = dict(features_dict)
    if "county_mean_coverage" not in features_dict and "provincia" in features_dict:
        key = _normalize(features_dict["provincia"])
        features_dict["county_mean_coverage"] = county_mean_lookup.get(key, 0.0)
        if key not in county_mean_lookup:
            print(f"  Warning: '{features_dict['provincia']}' not in lookup")

    row    = {col: features_dict.get(col, 0) for col in feature_cols}
    X_pred = pd.DataFrame([row])

    def _sig(x): return float(1 / (1 + np.exp(-x)))

    pred_lgbm = _sig(lgbm.predict(X_pred)[0])
    pred_rf   = _sig(rf.predict(X_pred)[0])
    pred_xgb  = _sig(xgb.predict(X_pred)[0])

    total = sum(cv_scores.values()) if cv_scores else 0
    if total > 0:
        w = {k: v / total for k, v in cv_scores.items()}
        ensemble = pred_lgbm * w["LightGBM"] + pred_rf * w["RandomForest"] + pred_xgb * w["XGBoost"]
    else:
        ensemble = float(np.mean([pred_lgbm, pred_rf, pred_xgb]))

    print(f"\n--- Prediction ---")
    for k, v in features_dict.items():
        print(f"  {k}: {v}")
    print(f"  LightGBM:     {pred_lgbm:.1%}")
    print(f"  RandomForest: {pred_rf:.1%}")
    print(f"  XGBoost:      {pred_xgb:.1%}")
    print(f"  Ensemble:     {ensemble:.1%}")

    return {"lgbm": pred_lgbm, "rf": pred_rf, "xgb": pred_xgb, "ensemble": ensemble}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("Loading insurance coverage data...")
    df_ins = load_insurance_data()

    print("\nLoading NUTS3 province features from DB...")
    df_nuts3 = load_nuts3_features()

    print("\nLoading building counts from DB...")
    df_buildings = load_building_counts()

    print("\nLoading comune coordinates (geocoded or cached)...")
    df_coords = load_comune_coordinates(df_ins)

    print("\nLoading waterway distances...")
    df_coords = load_waterway_features(df_coords)

    print("\nLoading unemployment data (ISTAT)...")
    df_ins = load_unemployment_data(df_ins)

    print("\nJoining features...")
    df = build_features(df_ins, df_nuts3, df_buildings, df_coords)

    print("\nPreparing model data...")
    X, y, y_orig, df_model = prepare_model_data(df)

    print(f"\nComuni used: {len(df_model):,}")
    print(df_model[TARGET_COL].describe().round(4))

    print("\nTraining models (5-fold CV)...\n")
    lgbm, rf, xgb, cv_preds, cv_scores = train_models(X, y, y_orig)

    print("\nSaving model bundle...")
    county_mean_lookup = df_model.groupby("provincia_key")["coverage_rate"].mean().to_dict()
    save_models(lgbm, rf, xgb, cv_scores, list(X.columns), county_mean_lookup)

    # Example prediction — update with real values once you have data
    print("\nExample prediction...")
    predict_from_features({
        "provincia":          "Milano",
        "tip_code":           1,
        "log_n_abitazioni":   np.log1p(15_000),
        "loc_lat":            45.46,
        "loc_lon":            9.19,
        "dist_to_waterway_km": 1.5,
    })

# ==============================================================
# Locality-Level PAD Insurance Coverage Analysis
# Target : coverage_rate (n_pad / n_locuinte per locality)
# Sources: UAT xlsx (~3,000 localities) + DB (NUTS3 + footprints)
# ==============================================================

import argparse
import copy
import joblib
import os
import re
import time
import unicodedata
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import shap
import yaml
from dotenv import load_dotenv
from scipy.stats import spearmanr
from sqlalchemy import create_engine, text
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

# =========================
# CONFIG
# =========================

# Load secrets from .env (no-op if already set in environment)
load_dotenv(Path(__file__).parent / ".env")

# Anchor all relative paths to the directory containing this script
BASE_DIR = Path(__file__).parent

DB_CONFIG = dict(
    host=os.getenv("DB_HOST", "10.100.0.10"),
    dbname=os.getenv("DB_NAME", "riskdata"),
    user=os.getenv("DB_USER", "geoserver"),
    password=os.getenv("DB_PASSWORD", ""),
    port=int(os.getenv("DB_PORT", "31183")),
)

XLSX_PATH    = BASE_DIR / "data" / "input" / "UAT-Grad-cuprindere-in-asigurare-la-31-01-2026.xlsx"
COUNTRY_CODE = "RO"
TARGET_COL      = "coverage_rate"
AVG_PREMIUM_EUR = 20.0   # PAD Type A = 20 EUR/yr, Type B = 10 EUR/yr

# Geocoding (Google Maps API — runs once, result cached to CSV)
GOOGLE_API_KEY        = os.getenv("GOOGLE_API_KEY", "")
LOCALITY_COORDS_CACHE = BASE_DIR / "data" / "cache" / "locality_coords_ro.csv"

# Waterway lines (OSM HOT export — LineString GeoJSON)
WATERWAY_PATH  = BASE_DIR / "data" / "input" / "hotosm_rou_waterways_lines.geojson"
WATERWAY_CACHE = BASE_DIR / "data" / "cache" / "locality_waterway_dist_ro.csv"
WATERWAY_TYPES = ("river", "canal", "stream")   # exclude drain/ditch (minor features)

# Building counts cache (heavy spatial join — run once and reuse)
BUILDING_COUNTS_CACHE = BASE_DIR / "data" / "cache" / "building_counts_ro.csv"

# Unemployment data (ANOFM PDF — parsed once and cached)
UNEMPLOYMENT_PDF   = BASE_DIR / "someri-localitati-ian-2026.pdf"
UNEMPLOYMENT_CACHE = BASE_DIR / "data" / "cache" / "unemployment_ro.csv"

# Ensure data directories exist
(BASE_DIR / "data" / "cache").mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Saved model path (output of save_models — used by predict_from_features)
MODEL_PATH = OUTPUT_DIR / "model.pkl"

# County drill-down
FOCUS_COUNTY = "Arad"   # county to analyse in detail

# =========================
# REINSURANCE EXTENSIONS
# =========================
SUM_INSURED_EUR        = 20_000   # statutory sum insured per PAD policy (EUR)
VULNERABILITY_FACTOR   = 0.30     # mean damage ratio (simplified, replace with real MDR curve)
MANDATORY_SCENARIO_RATE = 0.80    # 80% compliance enforcement scenario
BOOTSTRAP_ITERATIONS   = 100      # bootstrap samples for prediction uncertainty (~3 min)
CAT_STRESS_MULTIPLIER  = 3.0      # approximate 1-in-100 year loss multiplier


# =========================
# CLI / CONFIG
# =========================

def _parse_args():
    p = argparse.ArgumentParser(
        description="PAD insurance coverage model — train and evaluate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",        default=str(BASE_DIR / "config.yaml"),
                   help="Path to config.yaml")
    p.add_argument("--xlsx",          default=None,
                   help="Override input xlsx path")
    p.add_argument("--country",       default=None,
                   help="Override country code (e.g. RO, IT)")
    p.add_argument("--focus-county",  default=None, dest="focus_county",
                   help="County to highlight in drill-down plots")
    p.add_argument("--output-dir",    default=None, dest="output_dir",
                   help="Override output directory")
    p.add_argument("--no-plots",      action="store_true",
                   help="Skip all matplotlib figures (faster, headless-friendly)")
    p.add_argument("--no-bootstrap",  action="store_true",
                   help="Skip bootstrap prediction intervals (saves ~3 min)")
    return p.parse_args()


def _apply_config(args):
    """
    Load config.yaml and apply CLI overrides to module-level globals.
    Priority: CLI args > environment variables > config.yaml > hard-coded defaults.
    """
    global DB_CONFIG, XLSX_PATH, COUNTRY_CODE, GOOGLE_API_KEY
    global LOCALITY_COORDS_CACHE, WATERWAY_PATH, WATERWAY_CACHE, WATERWAY_TYPES
    global BUILDING_COUNTS_CACHE, UNEMPLOYMENT_PDF, UNEMPLOYMENT_CACHE
    global OUTPUT_DIR, MODEL_PATH, FOCUS_COUNTY
    global AVG_PREMIUM_EUR, SUM_INSURED_EUR, VULNERABILITY_FACTOR
    global MANDATORY_SCENARIO_RATE, BOOTSTRAP_ITERATIONS, CAT_STRESS_MULTIPLIER

    config_path = Path(args.config)
    cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    elif args.config != str(BASE_DIR / "config.yaml"):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # DB — env vars always win; config.yaml provides defaults
    db_cfg = cfg.get("database", {})
    DB_CONFIG = dict(
        host    = os.getenv("DB_HOST",     str(db_cfg.get("host",     DB_CONFIG["host"]))),
        dbname  = os.getenv("DB_NAME",     str(db_cfg.get("name",     DB_CONFIG["dbname"]))),
        user    = os.getenv("DB_USER",     str(db_cfg.get("user",     DB_CONFIG["user"]))),
        password= os.getenv("DB_PASSWORD", str(db_cfg.get("password", DB_CONFIG["password"]))),
        port    = int(os.getenv("DB_PORT", str(db_cfg.get("port",     DB_CONFIG["port"])))),
    )

    # Paths
    paths = cfg.get("paths", {})
    if args.xlsx:
        XLSX_PATH = Path(args.xlsx)
    elif "xlsx" in paths:
        XLSX_PATH = BASE_DIR / paths["xlsx"]

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    elif "output_dir" in paths:
        OUTPUT_DIR = BASE_DIR / paths["output_dir"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = OUTPUT_DIR / "model.pkl"

    if "waterway_geojson" in paths:
        WATERWAY_PATH = BASE_DIR / paths["waterway_geojson"]
    if "unemployment_pdf" in paths:
        UNEMPLOYMENT_PDF = BASE_DIR / paths["unemployment_pdf"]

    # Cache paths
    cache = cfg.get("cache", {})
    if "locality_coords"  in cache: LOCALITY_COORDS_CACHE = BASE_DIR / cache["locality_coords"]
    if "waterway_dist"    in cache: WATERWAY_CACHE        = BASE_DIR / cache["waterway_dist"]
    if "building_counts"  in cache: BUILDING_COUNTS_CACHE = BASE_DIR / cache["building_counts"]
    if "unemployment"     in cache: UNEMPLOYMENT_CACHE    = BASE_DIR / cache["unemployment"]
    (BASE_DIR / "data" / "cache").mkdir(parents=True, exist_ok=True)

    # Waterway types
    ww = cfg.get("waterway", {})
    if "types" in ww:
        WATERWAY_TYPES = tuple(ww["types"])

    # Country / focus county
    COUNTRY_CODE = args.country or cfg.get("country_code", COUNTRY_CODE)
    FOCUS_COUNTY = args.focus_county or cfg.get("focus_county", FOCUS_COUNTY)

    # Google API key
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", cfg.get("google_api_key", GOOGLE_API_KEY))

    # Model params
    mdl = cfg.get("model", {})
    if "bootstrap_iterations" in mdl:
        BOOTSTRAP_ITERATIONS = 0 if args.no_bootstrap else int(mdl["bootstrap_iterations"])
    elif args.no_bootstrap:
        BOOTSTRAP_ITERATIONS = 0

    # Business constants
    biz = cfg.get("business", {})
    if "avg_premium_eur"        in biz: AVG_PREMIUM_EUR         = float(biz["avg_premium_eur"])
    if "sum_insured_eur"        in biz: SUM_INSURED_EUR         = float(biz["sum_insured_eur"])
    if "vulnerability_factor"   in biz: VULNERABILITY_FACTOR    = float(biz["vulnerability_factor"])
    if "mandatory_scenario_rate"in biz: MANDATORY_SCENARIO_RATE = float(biz["mandatory_scenario_rate"])
    if "cat_stress_multiplier"  in biz: CAT_STRESS_MULTIPLIER   = float(biz["cat_stress_multiplier"])


# =========================
# 0. HELPERS
# =========================

def _normalize(s):
    """Strip accents and uppercase for name matching."""
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.strip().upper()

def _make_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}",
        connect_args={
            "keepalives": 1,
            "keepalives_idle": 5,       # send keepalive after 5 s idle (prevents SSL drop on long queries)
            "keepalives_interval": 3,
            "keepalives_count": 10,
            "connect_timeout": 120,
        }
    )


# =========================
# 1. LOAD DATA
# =========================

def load_xlsx_data(path=XLSX_PATH):
    """Load locality-level PAD coverage from xlsx."""
    df = pd.read_excel(path, header=1)
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
    df["coverage_rate"] = (df["n_pad"] / df["n_locuinte"]).fillna(0).clip(0, 1)
    df["judet_key"] = df["judet"].apply(_normalize)
    print(f"  Localities: {len(df):,}  |  Counties: {df['judet'].nunique()}")
    return df


def load_nuts3_features(country_code=COUNTRY_CODE):
    """
    Query NUTS3 table for structural county attributes + centroid lat/lon.
    Returns one row per county.
    """
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
    df["judet_key"] = df["name_latn"].apply(_normalize)
    print(f"  NUTS3 counties loaded: {len(df)}")
    return df


def load_building_counts(country_code=COUNTRY_CODE, cache_path=BUILDING_COUNTS_CACHE):
    """
    Query total buildings + insured value per county per peril from footprints.
    Result is cached to CSV — the spatial join is very slow, so we only run it once.
    Returns one row per county with n_buildings_*, avg_insured_net_*, retention_ratio_*,
    log_avg_footprint_m2, and other derived composite features.
    """
    if os.path.exists(cache_path):
        pivot = pd.read_csv(cache_path)
        print(f"  Loaded building counts from cache ({len(pivot)} counties)")
        return pivot

    print("  No cache found — running spatial join (this will take several minutes)...")
    engine = _make_engine()

    try:
        query = text("""
            SELECT
                n.name_latn AS county_name,
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
    except Exception as e:
        raise RuntimeError(
            f"Building counts spatial query failed: {e}\n"
            "Fix the DB connection or provide building_counts_ro.csv manually.\n"
            "Building features (n_buildings_*, retention_ratio_*, log_avg_footprint_m2) "
            "are required for meaningful predictions."
        ) from e

    for col in ["n_buildings", "sum_insured_net", "sum_insured_gross",
                "avg_insured_net"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Pivot counts
    pivot_count = df.pivot_table(
        index="county_name", columns="covered_peril",
        values="n_buildings", aggfunc="sum", fill_value=0,
    )
    pivot_count.columns = [f"n_buildings_{c.lower()}" for c in pivot_count.columns]

    # Pivot sum insured net
    pivot_net = df.pivot_table(
        index="county_name", columns="covered_peril",
        values="sum_insured_net", aggfunc="sum", fill_value=0,
    )
    pivot_net.columns = [f"sum_insured_net_{c.lower()}" for c in pivot_net.columns]

    # Pivot sum insured gross
    pivot_gross = df.pivot_table(
        index="county_name", columns="covered_peril",
        values="sum_insured_gross", aggfunc="sum", fill_value=0,
    )
    pivot_gross.columns = [f"sum_insured_gross_{c.lower()}" for c in pivot_gross.columns]

    # Pivot avg insured net
    pivot_avg = df.pivot_table(
        index="county_name", columns="covered_peril",
        values="avg_insured_net", aggfunc="mean", fill_value=0,
    )
    pivot_avg.columns = [f"avg_insured_net_{c.lower()}" for c in pivot_avg.columns]

    # Pivot avg footprint area (mean across perils → one value per county)
    pivot_area = df.pivot_table(
        index="county_name", columns="covered_peril",
        values="avg_footprint_m2", aggfunc="mean", fill_value=0,
    )
    pivot_area.columns = [f"avg_footprint_m2_{c.lower()}" for c in pivot_area.columns]

    pivot = pivot_count.join(pivot_net).join(pivot_gross).join(pivot_avg).join(pivot_area).reset_index()

    # Derived: total building count + log
    n_cols = [c for c in pivot.columns if c.startswith("n_buildings")]
    pivot["total_buildings"]     = pivot[n_cols].sum(axis=1)
    pivot["log_total_buildings"] = np.log1p(pivot["total_buildings"])

    # Derived: retention ratio per peril (net / gross — lower = more reinsured = riskier county)
    for peril in ["flood", "earthquake", "fire"]:
        net_col   = f"sum_insured_net_{peril}"
        gross_col = f"sum_insured_gross_{peril}"
        if net_col in pivot.columns and gross_col in pivot.columns:
            pivot[f"retention_ratio_{peril}"] = np.where(
                pivot[gross_col] > 0,
                (pivot[net_col] / pivot[gross_col]).clip(0, 1),
                np.nan,
            )

    # Derived: average building footprint size across perils (m²) — property size proxy
    area_cols = [c for c in pivot.columns if c.startswith("avg_footprint_m2_")]
    if area_cols:
        pivot["avg_footprint_m2"] = pivot[area_cols].replace(0, np.nan).mean(axis=1)
        pivot["log_avg_footprint_m2"] = np.log1p(pivot["avg_footprint_m2"].fillna(0))

    pivot["judet_key"] = pivot["county_name"].apply(_normalize)

    pivot.to_csv(cache_path, index=False)
    print(f"  Building counts loaded for {len(pivot)} counties — saved to {cache_path}")
    return pivot


def load_locality_coordinates(df_xlsx, cache_path=LOCALITY_COORDS_CACHE):
    """
    Return locality-level lat/lon from cache CSV, or geocode via Google Maps API
    and save the cache for subsequent runs.
    Columns returned: judet, localitate, loc_lat, loc_lon
    """
    if os.path.exists(cache_path):
        df_coords = pd.read_csv(cache_path)
        n_ok = df_coords["loc_lat"].notna().sum()
        print(f"  Loaded locality coordinates from cache ({n_ok}/{len(df_coords)} geocoded)")
        return df_coords

    localities = df_xlsx[["judet", "localitate"]].drop_duplicates().reset_index(drop=True)
    print(f"  Geocoding {len(localities):,} localities via Google Maps API ...")

    rows = []
    for i, row in localities.iterrows():
        address = f"{row['localitate']}, {row['judet']}, Romania"
        loc_lat, loc_lon = None, None
        try:
            resp = requests.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": address, "key": GOOGLE_API_KEY, "region": "ro"},
                timeout=10,
            )
            data = resp.json()
            if data.get("status") == "OK" and data.get("results"):
                geo = data["results"][0]["geometry"]["location"]
                loc_lat, loc_lon = geo["lat"], geo["lng"]
        except Exception:
            pass
        rows.append({"judet": row["judet"], "localitate": row["localitate"],
                     "loc_lat": loc_lat, "loc_lon": loc_lon})
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(localities)} done ...")
        time.sleep(0.05)   # ~20 req/s, well within free-tier limits

    df_coords = pd.DataFrame(rows)
    df_coords.to_csv(cache_path, index=False)
    n_ok = df_coords["loc_lat"].notna().sum()
    print(f"  Geocoded {n_ok}/{len(df_coords)} localities — saved to {cache_path}")
    return df_coords


def load_waterway_features(
    df_coords,
    waterway_path=WATERWAY_PATH,
    cache_path=WATERWAY_CACHE,
    waterway_types=WATERWAY_TYPES,
):
    """
    Compute distance (km) from each geocoded locality to the nearest significant
    waterway line (river / canal / stream).  Result is cached to avoid re-running
    the spatial join on every script execution.

    Returns df_coords with an added 'dist_to_waterway_km' column.
    """
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path)
        df_coords = df_coords.merge(
            cache[["judet", "localitate", "dist_to_waterway_km"]],
            on=["judet", "localitate"], how="left",
        )
        n_ok = df_coords["dist_to_waterway_km"].notna().sum()
        print(f"  Loaded waterway distances from cache ({n_ok}/{len(df_coords)} localities)")
        return df_coords

    if not os.path.exists(waterway_path):
        print(f"  Waterway file not found: {waterway_path}  — skipping")
        return df_coords

    print(f"  Loading waterway lines from {waterway_path} ...")
    waterways = gpd.read_file(waterway_path)
    print(f"  Waterway type counts:\n{waterways['waterway'].value_counts().to_string()}")

    waterways = waterways[waterways["waterway"].isin(waterway_types)].copy()
    if waterways.empty:
        print(f"  No features matched types {waterway_types} — skipping")
        return df_coords

    # Project to metric CRS (EPSG:3844 = Stereo 70, Romanian national grid, units: metres)
    waterways = waterways[["geometry"]].to_crs("EPSG:3844")

    # Build locality GeoDataFrame from geocoded coords (drop any without coordinates)
    df_valid = df_coords.dropna(subset=["loc_lat", "loc_lon"]).reset_index(drop=True)
    gdf_loc = gpd.GeoDataFrame(
        df_valid[["judet", "localitate"]].copy(),
        geometry=gpd.points_from_xy(df_valid["loc_lon"], df_valid["loc_lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3844")

    print(f"  Computing nearest waterway distance for {len(gdf_loc):,} localities ...")
    nearest = gpd.sjoin_nearest(
        gdf_loc, waterways, how="left", distance_col="dist_m"
    )
    # sjoin_nearest can produce duplicates if equidistant — keep minimum per original index
    nearest = nearest.groupby(nearest.index)["dist_m"].min()
    # Use .map() to align by index, not positional .values (avoids silent misalignment)
    gdf_loc["dist_to_waterway_km"] = (gdf_loc.index.map(nearest) / 1000).round(3)

    # Merge back onto full df_coords (localities without coords get NaN → fillna later)
    df_coords = df_coords.merge(
        gdf_loc[["judet", "localitate", "dist_to_waterway_km"]],
        on=["judet", "localitate"], how="left",
    )

    # Cache result
    df_coords[["judet", "localitate", "dist_to_waterway_km"]].to_csv(cache_path, index=False)
    n_ok = df_coords["dist_to_waterway_km"].notna().sum()
    print(f"  Waterway distances computed for {n_ok}/{len(df_coords)} localities")
    print(f"  Median distance: {df_coords['dist_to_waterway_km'].median():.2f} km  "
          f"(min: {df_coords['dist_to_waterway_km'].min():.2f}  "
          f"max: {df_coords['dist_to_waterway_km'].max():.2f})")
    print(f"  Saved to {cache_path}")
    return df_coords


def load_unemployment_data(df_xlsx, pdf_path=UNEMPLOYMENT_PDF, cache_path=UNEMPLOYMENT_CACHE):
    """
    Parse the ANOFM locality-level unemployment PDF (someri-localitati-*.pdf).
    Returns df_xlsx with an added 'someri_rate' column (registered unemployed / n_locuinte).
    Result is cached to CSV after the first parse.

    Join strategy:
      1. Strip locality-type prefixes (MUNICIPIUL, ORAS, COMUNA, MUN.) and normalize.
      2. Second pass: normalize hyphens → spaces to catch e.g. MIERCUREA-CIUC / POPESTI-LEORDENI.
      3. Remaining unmatched localities get county-median imputation.
    """
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
        n_ok = cache["someri"].notna().sum()
        print(f"  Loaded unemployment data from cache ({n_ok}/{len(cache)} localities matched)")
        return df_xlsx.merge(cache[["judet", "localitate", "someri"]], on=["judet", "localitate"], how="left")

    if not pdf_path.exists():
        print(f"  Unemployment PDF not found: {pdf_path} — skipping")
        return df_xlsx

    try:
        import pdfplumber
    except ImportError:
        print("  pdfplumber not installed — run: pip install pdfplumber")
        return df_xlsx

    def _strip_prefix(s):
        s = _normalize(s)
        s = re.sub(r"^MUN\.\s*", "", s)
        s = re.sub(r"^(MUNICIPIUL|ORAS|COMUNA)\s+", "", s)
        return s

    def _dehyphen(s):
        return s.replace("-", " ")

    def _parse_int(s):
        if s is None:
            return None
        try:
            return int(str(s).replace(".", "").replace(",", ".").strip())
        except ValueError:
            return None

    # ── Parse PDF ────────────────────────────────────────────────────────────
    print(f"  Parsing unemployment PDF ({pdf_path.name}) ...")
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if not table:
                continue
            for row in table:
                if len(row) < 7:
                    continue
                judet, mediu, localitate, _, _, total, _ = row[:7]
                if mediu is None or str(mediu).strip().lower() in (
                    "mediu\nurban/rural", "tara", "judet", ""
                ):
                    continue
                n = _parse_int(total)
                if n is None or judet is None or localitate is None:
                    continue
                rows.append({
                    "judet_key":  _normalize(judet),
                    "loc_key":    _strip_prefix(localitate),
                    "loc_key_nh": _dehyphen(_strip_prefix(localitate)),
                    "someri":     n,
                })
    df_pdf = pd.DataFrame(rows)

    # ── Build join keys on xlsx side ─────────────────────────────────────────
    xlsx = df_xlsx[["judet", "localitate"]].drop_duplicates().copy()
    xlsx["judet_key"]    = xlsx["judet"].apply(_normalize)
    xlsx["loc_key_xlsx"] = xlsx["localitate"].apply(_normalize)
    xlsx["loc_key_nh"]   = xlsx["loc_key_xlsx"].apply(lambda s: s.replace("-", " "))

    # Pass 1: exact normalized match
    merged = xlsx.merge(
        df_pdf[["judet_key", "loc_key", "someri"]],
        left_on=["judet_key", "loc_key_xlsx"],
        right_on=["judet_key", "loc_key"],
        how="left",
    ).drop(columns=["loc_key"])

    # Pass 2: hyphen-normalized match for remaining unmatched rows
    unmatched_mask = merged["someri"].isna()
    if unmatched_mask.any():
        fill = merged[unmatched_mask][["judet_key", "loc_key_nh"]].merge(
            df_pdf[["judet_key", "loc_key_nh", "someri"]],
            on=["judet_key", "loc_key_nh"],
            how="left",
        )["someri"]
        merged.loc[unmatched_mask, "someri"] = fill.values

    n_matched   = merged["someri"].notna().sum()
    n_total     = len(merged)
    n_unmatched = n_total - n_matched
    print(f"  Matched: {n_matched}/{n_total} localities "
          f"({n_matched/n_total:.1%}){f'  — {n_unmatched} get county-median' if n_unmatched else ''}")

    # Cache the locality-level matched table (judet, localitate, someri)
    out = merged[["judet", "localitate", "someri"]].copy()
    out.to_csv(cache_path, index=False)

    return df_xlsx.merge(out, on=["judet", "localitate"], how="left")


# =========================
# 2. FEATURE ENGINEERING
# =========================

def build_features(df_xlsx, df_nuts3, df_buildings, df_coords=None):
    """
    Join NUTS3, building, geocoded coordinates, and (optionally) unemployment
    data onto locality-level xlsx data.
    """
    df = df_xlsx.copy()

    # Settlement type: extract leading digit
    df["tip_code"] = (
        df["tip"].astype(str).str.extract(r"^(\d)")[0].astype(float)
    )
    df["log_n_locuinte"] = np.log1p(df["n_locuinte"])

    # Public housing insured rate
    n_pub = pd.to_numeric(df["n_locuinte_public"], errors="coerce")
    n_pad_pub = pd.to_numeric(df["n_pad_public"], errors="coerce")
    df["coverage_rate_public"] = (n_pad_pub / n_pub).fillna(0).clip(0, 1)

    # Join NUTS3 structural attributes + county centroid (county-level)
    nuts3_cols = ["judet_key", "urbn_type", "mount_type", "coast_type", "lat", "lon"]
    df = df.merge(df_nuts3[nuts3_cols], on="judet_key", how="left")

    # Join building counts + insured values (county-level)
    # Guard: if the query returned no rows (e.g. DB timeout), skip and fill zeros later
    bld_base = ["judet_key", "log_total_buildings",
                "n_buildings_flood", "n_buildings_fire", "n_buildings_earthquake"]
    bld_base_available = [c for c in bld_base if c in df_buildings.columns]
    bld_extra = [c for c in df_buildings.columns
                 if c in ("avg_insured_net_flood", "avg_insured_net_fire",
                          "retention_ratio_flood", "retention_ratio_earthquake",
                          "retention_ratio_fire", "log_avg_footprint_m2")]
    if bld_base_available and len(df_buildings) > 0:
        df = df.merge(df_buildings[bld_base_available + bld_extra], on="judet_key", how="left")
    else:
        print("  Warning: building data unavailable — building features will be zero-filled")
        for col in bld_base[1:] + bld_extra:   # skip judet_key
            df[col] = 0.0

    # Join locality-level coordinates (and waterway distance if present)
    if df_coords is not None:
        coord_cols = ["judet", "localitate", "loc_lat", "loc_lon"]
        if "dist_to_waterway_km" in df_coords.columns:
            coord_cols.append("dist_to_waterway_km")
        df = df.merge(df_coords[coord_cols], on=["judet", "localitate"], how="left")
        # Fall back to county centroid for any unmatched locality
        df["loc_lat"] = df["loc_lat"].fillna(df["lat"])
        df["loc_lon"] = df["loc_lon"].fillna(df["lon"])
        n_fallback = df["loc_lat"].isna().sum()
        if n_fallback:
            print(f"  Warning: {n_fallback} localities missing coordinates entirely")
        # Fill missing waterway distances with dataset median (unmatched localities)
        if "dist_to_waterway_km" in df.columns:
            median_dist = df["dist_to_waterway_km"].median()
            df["dist_to_waterway_km"] = df["dist_to_waterway_km"].fillna(median_dist)
    else:
        df["loc_lat"] = df["lat"]
        df["loc_lon"] = df["lon"]

    # Count of peril types with at least one building exposed
    df["n_active_perils"] = (
        (df["n_buildings_flood"].fillna(0)      > 0).astype(int) +
        (df["n_buildings_earthquake"].fillna(0) > 0).astype(int) +
        (df["n_buildings_fire"].fillna(0)       > 0).astype(int)
    )

    # Composite hazard intensity proxy (normalized 0–1)
    # Replace weights with real flood depth / PGA values when available
    df["hazard_intensity"] = (
        df["n_buildings_flood"].fillna(0)      * 0.4 +
        df["n_buildings_earthquake"].fillna(0) * 0.4 +
        df["n_buildings_fire"].fillna(0)       * 0.2
    )
    max_hi = df["hazard_intensity"].max()
    if max_hi > 0:
        df["hazard_intensity"] /= max_hi

    n_unmatched = df["urbn_type"].isna().sum()
    if n_unmatched:
        print(f"  Warning: {n_unmatched} localities did not match NUTS3 data")

    # Unemployment rate: someri / n_locuinte (locality economic distress proxy)
    # County-median imputation for the ~10 localities not matched in the PDF.
    if "someri" in df.columns:
        df["someri_rate"] = (df["someri"] / df["n_locuinte"].replace(0, np.nan)).clip(0, None)
        county_median = df.groupby("judet_key")["someri_rate"].transform("median")
        df["someri_rate"] = df["someri_rate"].fillna(county_median)
        df["someri_rate"] = df["someri_rate"].fillna(df["someri_rate"].median())

    # County-mean coverage (leave-one-out: excludes the locality itself to prevent
    # target leakage in cross-validation).
    county_sum   = df.groupby("judet_key")["coverage_rate"].transform("sum")
    county_count = df.groupby("judet_key")["coverage_rate"].transform("count")
    df["county_mean_coverage"] = (county_sum - df["coverage_rate"]) / (county_count - 1)
    df["county_mean_coverage"] = df["county_mean_coverage"].fillna(df["coverage_rate"])

    return df


# =========================
# 3. MODEL PREP
# =========================

FEATURE_COLS = [
    # From xlsx
    "tip_code",           # 1=Municipiu, 2=Oras, 3=Comuna
    "log_n_locuinte",     # log housing stock
    "coverage_rate_public",    # public housing insured rate (n_pad_public / n_locuinte_public)
    # From NUTS3
    "urbn_type",          # 1=urban … 3=rural
    "mount_type",         # 1=mountain … 4=non-mountain
    "coast_type",         # 1=coastal, 2=non-coastal
    "loc_lat",            # locality latitude  (north/south; falls back to county centroid)
    "loc_lon",            # locality longitude (east/west;  falls back to county centroid)
    # From footprints — counts
    "log_total_buildings",
    "n_buildings_flood",
    "n_buildings_fire",
    "n_buildings_earthquake",
    "n_active_perils",              # count of peril types with n_buildings > 0 (0–3)
    # From footprints — avg insured value per building (wealth proxy)
    "avg_insured_net_flood",        # ρ=−0.143 **
    "avg_insured_net_fire",         # ρ=−0.129 **
    # From footprints — reinsurance retention ratio (net/gross; lower = riskier county)
    "retention_ratio_flood",
    "retention_ratio_earthquake",
    "retention_ratio_fire",
    # From footprints — average building footprint area (property size proxy)
    "log_avg_footprint_m2",
    "hazard_intensity",        # composite risk proxy (flood×0.4 + quake×0.4 + fire×0.2), normalised
    "dist_to_waterway_km",     # locality-level: distance to nearest river/canal/stream (km)
    # From ANOFM unemployment PDF
    "someri_rate",             # registered unemployed / n_locuinte (local economic distress)
    # Derived county-level aggregate
    "county_mean_coverage",    # LOO county mean of coverage_rate (county fixed effect)
]

def prepare_model_data(df):
    df = df[df[TARGET_COL].notna()].copy().reset_index(drop=True)
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].copy()

    # Per-feature imputation: median for continuous/ratio features, 0 for counts and flags.
    # Imputing 0 for log/ratio features is semantically wrong (e.g. log_avg_footprint_m2=0
    # implies 1 m² buildings; retention_ratio=0 implies fully ceded).
    _MEDIAN_FILL = {
        "avg_insured_net_flood", "avg_insured_net_fire",
        "retention_ratio_flood", "retention_ratio_earthquake", "retention_ratio_fire",
        "log_avg_footprint_m2", "dist_to_waterway_km", "someri_rate",
    }
    for col in X.columns:
        if col in _MEDIAN_FILL:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(0)

    y_orig = df[TARGET_COL].values
    # Clip before logit: coverage_rate = 0 or 1 exactly causes log(0) or log(inf)
    eps = 1e-6
    y_safe = np.clip(y_orig, eps, 1 - eps)
    y = np.log(y_safe / (1 - y_safe))   # logit transform → unbounded target
    return X, y, y_orig, df


# =========================
# 4. TRAINING (5-fold CV)
# =========================

def train_models(X, y, y_orig):
    """
    Train on logit-transformed y; back-transform OOF predictions to coverage_rate
    space before computing metrics and returning cv_preds.
    Returns lgbm, rf, xgb, cv_preds, cv_scores (R² per model for weighted ensemble).
    """
    # Stratify folds by coverage_rate quintile so each fold has a balanced
    # representation of low/high coverage localities (plain KFold can cluster
    # all high-coverage localities in one fold on skewed distributions).
    y_bins = pd.qcut(y_orig, q=5, labels=False, duplicates="drop")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(kf.split(X, y_bins))

    lgbm = LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=20, random_state=42, verbose=-1
    )
    rf = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=10, random_state=42, n_jobs=-1
    )
    xgb = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, tree_method="hist"
    )

    cv_preds = {}
    cv_scores = {}
    for name, model in [("LightGBM", lgbm), ("RandomForest", rf), ("XGBoost", xgb)]:
        y_pred_logit = cross_val_predict(model, X, y, cv=cv_splits)
        y_pred = 1 / (1 + np.exp(-y_pred_logit))   # sigmoid back-transform → [0, 1]
        cv_preds[name] = y_pred
        r2   = r2_score(y_orig, y_pred)
        mae  = mean_absolute_error(y_orig, y_pred)
        rmse = np.sqrt(mean_squared_error(y_orig, y_pred))
        cv_scores[name] = max(r2, 0.0)   # floor at 0 to avoid negative weights
        print(f"{name}")
        print(f"  R²:   {r2:.3f}")
        print(f"  MAE:  {mae:.4f}  ({mae*100:.2f} pp)")
        print(f"  RMSE: {rmse:.4f}  ({rmse*100:.2f} pp)\n")

    lgbm.fit(X, y)
    rf.fit(X, y)
    xgb.fit(X, y)
    return lgbm, rf, xgb, cv_preds, cv_scores


# =========================
# 5. ANALYSIS & PLOTS
# =========================

def plot_correlations(X, y):
    records = []
    for col in X.columns:
        rho, pval = spearmanr(X[col], y)
        records.append({"feature": col, "spearman_r": rho, "p_value": pval})
    corr = (
        pd.DataFrame(records)
        .sort_values("spearman_r", key=abs, ascending=False)
        .reset_index(drop=True)
    )
    corr["sig"] = corr["p_value"].apply(
        lambda p: "**" if p < 0.01 else ("*" if p < 0.05 else "")
    )
    print(f"\n--- Spearman correlations with {TARGET_COL} ---")
    print(corr.to_string(index=False))

    _, ax = plt.subplots(figsize=(8, max(4, len(X.columns) * 0.45)))
    colors = ["#e74c3c" if r > 0 else "#3498db" for r in corr["spearman_r"]]
    ax.barh(corr["feature"], corr["spearman_r"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spearman ρ")
    ax.set_title(f"Feature correlation with {TARGET_COL}\n(red=positive, blue=negative)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlations.png", dpi=150)
    plt.show()
    print(f"  Saved {OUTPUT_DIR / 'correlations.png'}")


def plot_coverage_by_type(df):
    tip_labels = {1: "Municipiu", 2: "Oras", 3: "Comuna"}
    df = df.copy()
    df["tip_label"] = df["tip_code"].map(tip_labels)
    _, ax = plt.subplots(figsize=(8, 5))
    groups = [
        df.loc[df["tip_label"] == label, TARGET_COL].dropna()
        for label in ["Municipiu", "Oras", "Comuna"]
    ]
    ax.boxplot(groups, tick_labels=["Municipiu", "Oras", "Comuna"], patch_artist=True)
    ax.set_ylabel("PAD coverage rate")
    ax.set_title("PAD coverage rate by settlement type")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "coverage_by_type.png", dpi=150)
    plt.show()
    print(f"  Saved {OUTPUT_DIR / 'coverage_by_type.png'}")


def plot_feature_importance(lgbm_model, rf_model, feature_names):
    lgbm_imp = pd.Series(lgbm_model.feature_importances_, index=feature_names).sort_values()
    rf_imp   = pd.Series(rf_model.feature_importances_,   index=feature_names).sort_values()
    _, axes = plt.subplots(1, 2, figsize=(13, max(4, len(feature_names) * 0.45)))
    lgbm_imp.plot(kind="barh", ax=axes[0], color="#e67e22")
    axes[0].set_title("LightGBM — gain importance")
    rf_imp.plot(kind="barh", ax=axes[1], color="#27ae60")
    axes[1].set_title("Random Forest — MDI importance")
    plt.suptitle(f"Feature importance  |  target: {TARGET_COL}", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.show()
    print(f"  Saved {OUTPUT_DIR / 'feature_importance.png'}")


def shap_analysis(model, X, top_n=4):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=list(X.columns),
                      max_display=10, show=False)
    plt.title(f"SHAP summary  |  target: {TARGET_COL}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150)
    plt.show()
    print(f"  Saved {OUTPUT_DIR / 'shap_summary.png'}")

    mean_abs     = np.abs(shap_values).mean(axis=0)
    top_features = pd.Series(mean_abs, index=X.columns).nlargest(top_n).index.tolist()
    for feat in top_features:
        _, ax = plt.subplots(figsize=(6, 4))
        shap.dependence_plot(feat, shap_values, X,
                             interaction_index="auto", ax=ax, show=False)
        ax.set_title(f"SHAP dependence: {feat}")
        plt.tight_layout()
        fname = OUTPUT_DIR / f"shap_dep_{feat}.png"
        plt.savefig(fname, dpi=150)
        plt.show()
        print(f"  Saved {fname}")


# =========================
# 6. SIMULATION HELPERS
# =========================

def simulate_expected_loss(df, tsi_column):
    """EL = TSI × hazard_intensity × vulnerability_factor, summed across localities."""
    return (df[tsi_column] * df["hazard_intensity"] * VULNERABILITY_FACTOR).sum()


def bootstrap_prediction_interval(model, X, y, n_iter=BOOTSTRAP_ITERATIONS):
    """
    Parametric bootstrap on a logit-space model.
    Returns (lower_5pct, upper_95pct) coverage_rate arrays aligned to X.
    Uses a deep copy so the fitted model is not modified.
    Runtime: ~2–4 min for n_iter=100 with LightGBM.
    """
    print(f"  Running {n_iter} bootstrap iterations ...")
    model_copy = copy.deepcopy(model)
    preds = []
    n = len(X)
    for i in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        model_copy.fit(X.iloc[idx], y[idx])
        pred_logit = model_copy.predict(X)
        preds.append(1 / (1 + np.exp(-pred_logit)))   # sigmoid back-transform
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{n_iter} done ...")
    preds = np.array(preds)
    lower = np.percentile(preds, 5,  axis=0)
    upper = np.percentile(preds, 95, axis=0)
    print(f"  Bootstrap done  |  median interval width: {(upper - lower).mean():.3f}")
    return lower, upper


# =========================
# 7. PREMIUM POTENTIAL
# =========================

def compute_premium_potential(df_model, cv_preds, cv_scores=None):
    """
    Estimate expected annual premium and untapped market opportunity per locality.
    Uses out-of-fold CV predictions (unbiased) as the model's coverage rate estimate.
    Saves premium_potential.csv, county_summary.csv, and premium_potential.png.
    If cv_scores is provided, uses R²-weighted ensemble; otherwise falls back to equal weights.
    """
    df = df_model.copy().reset_index(drop=True)

    # R²-weighted ensemble (softmax-style normalisation over non-negative R² scores)
    if cv_scores:
        total = sum(cv_scores.values())
        weights = {k: v / total for k, v in cv_scores.items()} if total > 0 else {k: 1/len(cv_scores) for k in cv_scores}
    else:
        weights = {k: 1 / len(cv_preds) for k in cv_preds}
    pred_rate = np.clip(
        sum(cv_preds[k] * weights[k] for k in cv_preds), 0, 1
    )
    df["predicted_coverage_rate"] = pred_rate

    # Premium metrics (EUR)
    df["current_premium_eur"]      = df["n_pad"] * AVG_PREMIUM_EUR
    df["predicted_premium_eur"]    = df["n_locuinte"] * df["predicted_coverage_rate"] * AVG_PREMIUM_EUR
    df["addressable_market_eur"]   = df["n_locuinte"] * AVG_PREMIUM_EUR
    # Opportunity = gap between model-predicted rate and actual rate (floored at 0)
    df["opportunity_eur"] = (
        (df["predicted_coverage_rate"] - df["coverage_rate"]).clip(lower=0)
        * df["n_locuinte"] * AVG_PREMIUM_EUR
    )

    # County-level summary
    county = (
        df.groupby("judet", sort=False)
        .agg(
            n_localities        = ("localitate", "count"),
            n_locuinte          = ("n_locuinte", "sum"),
            actual_coverage     = ("coverage_rate", "mean"),
            predicted_coverage  = ("predicted_coverage_rate", "mean"),
            current_premium_eur = ("current_premium_eur", "sum"),
            opportunity_eur     = ("opportunity_eur", "sum"),
            addressable_market_eur = ("addressable_market_eur", "sum"),
        )
        .sort_values("opportunity_eur", ascending=False)
        .reset_index()
    )

    # Print table
    print(f"\n--- Annual Premium Potential  (@{AVG_PREMIUM_EUR:.0f} EUR/policy) ---")
    hdr = f"{'County':<25} {'Locs':>6} {'Act%':>6} {'Pred%':>6} "
    hdr += f"{'Current (EUR)':>14} {'Opportunity (EUR)':>18} {'Total Market (EUR)':>18}"
    print(hdr)
    print("-" * len(hdr))
    for _, row in county.iterrows():
        print(
            f"{row['judet']:<25} {row['n_localities']:>6,} "
            f"{row['actual_coverage']:>5.1%} {row['predicted_coverage']:>6.1%} "
            f"{row['current_premium_eur']:>14,.0f} "
            f"{row['opportunity_eur']:>18,.0f} "
            f"{row['addressable_market_eur']:>18,.0f}"
        )
    print("-" * len(hdr))
    print(
        f"{'TOTAL':<25} {county['n_localities'].sum():>6,} "
        f"{'':>6} {'':>6} "
        f"{county['current_premium_eur'].sum():>14,.0f} "
        f"{county['opportunity_eur'].sum():>18,.0f} "
        f"{county['addressable_market_eur'].sum():>18,.0f}"
    )

    county.to_csv(OUTPUT_DIR / "county_summary.csv", index=False, float_format="%.4f")
    print(f"\n  Saved {OUTPUT_DIR / 'county_summary.csv'}")

    # === TSI (Total Sum Insured) ===
    df["tsi_actual"]    = df["n_pad"]      * SUM_INSURED_EUR
    df["tsi_predicted"] = df["n_locuinte"] * df["predicted_coverage_rate"] * SUM_INSURED_EUR
    df["tsi_mandatory"] = df["n_locuinte"] * MANDATORY_SCENARIO_RATE       * SUM_INSURED_EUR

    # === Expected Loss simulation ===
    if "hazard_intensity" in df.columns:
        loss_current   = simulate_expected_loss(df, "tsi_actual")
        loss_predicted = simulate_expected_loss(df, "tsi_predicted")
        loss_mandatory = simulate_expected_loss(df, "tsi_mandatory")
        loss_100yr     = loss_predicted * CAT_STRESS_MULTIPLIER

        print("\n--- Expected Loss Simulation ---")
        print(f"  Current Portfolio:      {loss_current:>15,.0f} EUR")
        print(f"  Structural Potential:   {loss_predicted:>15,.0f} EUR")
        print(f"  Mandatory Scenario:     {loss_mandatory:>15,.0f} EUR")
        print(f"  1-in-100 Stress Loss:   {loss_100yr:>15,.0f} EUR")

        # === Accumulation Index ===
        df["accumulation_index"] = df["tsi_predicted"] * df["hazard_intensity"]
        top_accum = (
            df.groupby("judet")["accumulation_index"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        print("\n--- Top 10 Accumulation Risk Counties ---")
        for judet_name, val in top_accum.items():
            print(f"  {judet_name:<25}  {val:>15,.0f} EUR")

        # === Exposure Concentration Shift ===
        total_actual = df["tsi_actual"].sum()
        total_pred   = df["tsi_predicted"].sum()
        county_shift = (
            df.groupby("judet")
            .agg(
                actual_share    = ("tsi_actual",    lambda x: x.sum() / total_actual),
                predicted_share = ("tsi_predicted", lambda x: x.sum() / total_pred),
            )
        )
        county_shift["shift_pp"] = (
            (county_shift["predicted_share"] - county_shift["actual_share"]) * 100
        )
        print("\n--- Exposure Concentration Shift (Predicted − Actual, pp) ---")
        for judet_name, row in county_shift.sort_values("shift_pp", ascending=False).head(10).iterrows():
            print(f"  {judet_name:<25}  {row['shift_pp']:>+.2f} pp")

    # Save locality-level CSV
    out_cols = [
        "judet", "tip", "localitate", "n_locuinte", "n_pad",
        "coverage_rate", "predicted_coverage_rate",
        "current_premium_eur", "opportunity_eur", "addressable_market_eur",
        "tsi_actual", "tsi_predicted",
    ]
    if "pred_lower" in df.columns:
        out_cols += ["pred_lower", "pred_upper"]
    if "accumulation_index" in df.columns:
        out_cols += ["accumulation_index"]
    (
        df[out_cols]
        .sort_values("opportunity_eur", ascending=False)
        .to_csv(OUTPUT_DIR / "premium_potential.csv", index=False, float_format="%.4f")
    )
    print(f"\n  Saved {OUTPUT_DIR / 'premium_potential.csv'}")

    # Bar chart: top 15 counties by opportunity
    top15 = county.head(15)
    _, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(top15))
    cur = top15["current_premium_eur"] / 1e6
    opp = top15["opportunity_eur"] / 1e6
    ax.bar(x, cur, label="Current premium", color="#3498db")
    ax.bar(x, opp, bottom=cur, label="Untapped opportunity", color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(top15["judet"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("EUR (millions)")
    ax.set_title(
        f"Expected annual PAD premium by county  (@{AVG_PREMIUM_EUR:.0f} EUR/policy)\n"
        f"Top 15 counties by untapped opportunity"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "premium_potential.png", dpi=150)
    plt.show()
    print(f"  Saved {OUTPUT_DIR / 'premium_potential.png'}")

    return df




def plot_county_focus(df_with_preds, county_name):
    """
    Drill into a single county: scatter actual vs predicted for all localities,
    colour by settlement type, and flag the biggest over/under-performers.
    """
    df_c = df_with_preds[
        df_with_preds["judet"].str.upper() == county_name.strip().upper()
    ].copy()

    if df_c.empty:
        print(f"  No data for county '{county_name}'. Check spelling.")
        return

    df_c["gap"] = df_c["predicted_coverage_rate"] - df_c["coverage_rate"]

    tip_colors = {1: "#e74c3c", 2: "#f39c12", 3: "#2ecc71"}
    tip_labels = {1: "Municipiu", 2: "Oras", 3: "Comuna"}

    _, ax = plt.subplots(figsize=(8, 7))
    for tip_val, grp in df_c.groupby("tip_code"):
        ax.scatter(
            grp["coverage_rate"], grp["predicted_coverage_rate"],
            c=tip_colors.get(tip_val, "gray"),
            label=tip_labels.get(tip_val, f"tip {tip_val}"),
            alpha=0.75, edgecolors="white", linewidth=0.3, s=60,
        )

    # Perfect-prediction diagonal
    lo = min(df_c["coverage_rate"].min(), df_c["predicted_coverage_rate"].min()) - 0.01
    hi = max(df_c["coverage_rate"].max(), df_c["predicted_coverage_rate"].max()) + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="Perfect prediction")

    # Annotate top 3 over-predicted and top 3 under-predicted
    for _, row in pd.concat([
        df_c.nlargest(3, "gap"),   # model over-estimates (opportunity)
        df_c.nsmallest(3, "gap"),  # model under-estimates (outperformers)
    ]).iterrows():
        ax.annotate(
            row["localitate"],
            (row["coverage_rate"], row["predicted_coverage_rate"]),
            fontsize=7, xytext=(5, 3), textcoords="offset points",
        )

    ax.set_xlabel("Actual coverage rate")
    ax.set_ylabel("Predicted coverage rate")
    ax.set_title(
        f"Actual vs Predicted PAD coverage — {county_name} county\n"
        f"({len(df_c)} localities,  above diagonal = model over-predicts = opportunity)"
    )
    fmt = plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.legend()
    plt.tight_layout()
    fname = OUTPUT_DIR / f"county_focus_{county_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  Saved {fname}")

    # Printed breakdown
    print(f"\nOutperformers (beating the model) in {county_name}:")
    for _, row in df_c.nsmallest(5, "gap").iterrows():
        print(f"  {row['localitate']:<30}  actual={row['coverage_rate']:.1%}  "
              f"pred={row['predicted_coverage_rate']:.1%}  gap={row['gap']:+.1%}")
    print(f"\nUnderperformers (below model expectation) in {county_name}:")
    for _, row in df_c.nlargest(5, "gap").iterrows():
        print(f"  {row['localitate']:<30}  actual={row['coverage_rate']:.1%}  "
              f"pred={row['predicted_coverage_rate']:.1%}  gap={row['gap']:+.1%}")


# =========================
# 8. PREDICTION
# =========================

def predict_locality(tip, judet, n_locuinte, df_nuts3, df_buildings, lgbm, rf, xgb, df_coords=None, cv_scores=None, county_mean_lookup=None):
    """
    Predict coverage_rate for a locality given its type, county, and housing stock.
    Returns a dict with per-model predictions and ensemble.
    """
    tip_code_map = {"MUNICIPIU": 1, "ORAS": 2, "COMUNA": 3}
    tip_code = tip_code_map.get(tip.upper())
    if tip_code is None:
        print(f"  Unknown tip '{tip}'. Use: MUNICIPIU, ORAS, COMUNA")
        return

    key = _normalize(judet)
    nuts3_row = df_nuts3[df_nuts3["judet_key"] == key]
    bld_row   = df_buildings[df_buildings["judet_key"] == key]

    if nuts3_row.empty:
        print(f"  County '{judet}' not found in NUTS3.")
        return

    # Compute hazard_intensity using same formula as build_features, normalised to full dataset
    if not bld_row.empty:
        n_flood = bld_row["n_buildings_flood"].values[0]
        n_quake = bld_row["n_buildings_earthquake"].values[0]
        n_fire  = bld_row["n_buildings_fire"].values[0]
        raw_hi = n_flood * 0.4 + n_quake * 0.4 + n_fire * 0.2
        raw_hi_all = (
            df_buildings["n_buildings_flood"].fillna(0)      * 0.4 +
            df_buildings["n_buildings_earthquake"].fillna(0) * 0.4 +
            df_buildings["n_buildings_fire"].fillna(0)       * 0.2
        )
        max_hi = raw_hi_all.max()
        hazard_intensity = float(raw_hi / max_hi) if max_hi > 0 else 0.0
        n_active_perils = int(n_flood > 0) + int(n_quake > 0) + int(n_fire > 0)
    else:
        hazard_intensity = 0.0
        n_active_perils = 0
        n_flood = n_quake = n_fire = 0

    def _bld(col):
        return float(bld_row[col].values[0]) if not bld_row.empty and col in bld_row.columns else 0.0

    # Use county centroid as loc_lat/loc_lon (reasonable for single-locality prediction)
    features = pd.DataFrame([{
        "tip_code":               tip_code,
        "log_n_locuinte":         np.log1p(n_locuinte),
        "coverage_rate_public":   0.0,   # unknown for ad-hoc prediction; use 0 as default
        "urbn_type":              nuts3_row["urbn_type"].values[0],
        "mount_type":             nuts3_row["mount_type"].values[0],
        "coast_type":             nuts3_row["coast_type"].values[0],
        "loc_lat":                nuts3_row["lat"].values[0],
        "loc_lon":                nuts3_row["lon"].values[0],
        "log_total_buildings":    _bld("log_total_buildings"),
        "n_buildings_flood":      n_flood,
        "n_buildings_fire":       n_fire,
        "n_buildings_earthquake": n_quake,
        "n_active_perils":        n_active_perils,
        "avg_insured_net_flood":        _bld("avg_insured_net_flood"),
        "avg_insured_net_fire":         _bld("avg_insured_net_fire"),
        "retention_ratio_flood":        _bld("retention_ratio_flood"),
        "retention_ratio_earthquake":   _bld("retention_ratio_earthquake"),
        "retention_ratio_fire":         _bld("retention_ratio_fire"),
        "log_avg_footprint_m2":         _bld("log_avg_footprint_m2"),
        "hazard_intensity":       hazard_intensity,
        "dist_to_waterway_km":    (
            df_coords.loc[df_coords["judet"] == judet, "dist_to_waterway_km"].median()
            if df_coords is not None and "dist_to_waterway_km" in df_coords.columns
            else 0.0
        ),
        "county_mean_coverage":   (
            county_mean_lookup.get(_normalize(judet), 0.0)
            if county_mean_lookup else 0.0
        ),
    }])

    cols = [c for c in FEATURE_COLS if c in features.columns]
    pred_lgbm = float(1 / (1 + np.exp(-lgbm.predict(features[cols])[0])))
    pred_rf   = float(1 / (1 + np.exp(-rf.predict(features[cols])[0])))
    pred_xgb  = float(1 / (1 + np.exp(-xgb.predict(features[cols])[0])))

    # R²-weighted ensemble — same formula as compute_premium_potential
    if cv_scores:
        total = sum(cv_scores.values())
        w = {k: v / total for k, v in cv_scores.items()} if total > 0 else {k: 1/3 for k in ["LightGBM", "RandomForest", "XGBoost"]}
        ensemble = pred_lgbm * w["LightGBM"] + pred_rf * w["RandomForest"] + pred_xgb * w["XGBoost"]
    else:
        ensemble = float(np.mean([pred_lgbm, pred_rf, pred_xgb]))

    print(f"\n--- Prediction: {tip} in {judet}, n_locuinte={n_locuinte:,} ---")
    print(f"  LightGBM:     {pred_lgbm:.1%}")
    print(f"  RandomForest: {pred_rf:.1%}")
    print(f"  XGBoost:      {pred_xgb:.1%}")
    print(f"  Ensemble:     {ensemble:.1%}")

    return {"lgbm": pred_lgbm, "rf": pred_rf, "xgb": pred_xgb, "ensemble": ensemble}


# =========================
# 9. MODEL PERSISTENCE
# =========================

def save_models(lgbm, rf, xgb, cv_scores, feature_cols, county_mean_lookup, path=MODEL_PATH):
    """
    Serialize the three trained models together with ensemble weights,
    feature column names, and county mean coverage lookup.
    The resulting file is fully self-contained —
    load it anywhere with joblib and call predict_from_features().
    """
    bundle = {
        "lgbm":                lgbm,
        "rf":                  rf,
        "xgb":                 xgb,
        "cv_scores":           cv_scores,
        "feature_cols":        feature_cols,
        "county_mean_lookup":  county_mean_lookup,  # judet_key → mean coverage_rate
    }
    joblib.dump(bundle, path)
    print(f"  Model bundle saved to {path}")


def predict_from_features(features_dict, model_path=MODEL_PATH):
    """
    Load a saved model bundle and predict coverage_rate for a single location.

    Parameters
    ----------
    features_dict : dict
        Any subset of the training features.  Unknown keys are ignored;
        missing features default to 0 (same as training fillna).
        Example::

            predict_from_features({
                "tip_code":           1,        # 1=Municipiu, 2=Oras, 3=Comuna
                "log_n_locuinte":     10.82,    # np.log1p(50_000)
                "loc_lat":            46.77,
                "loc_lon":            23.59,
                "dist_to_waterway_km": 2.1,
            })

    model_path : Path or str
        Path to the .pkl file produced by save_models().

    Returns
    -------
    dict with keys: lgbm, rf, xgb, ensemble  (all coverage_rate floats 0–1)
    """
    bundle = joblib.load(model_path)
    lgbm               = bundle["lgbm"]
    rf                 = bundle["rf"]
    xgb                = bundle["xgb"]
    cv_scores          = bundle["cv_scores"]
    feature_cols       = bundle["feature_cols"]
    county_mean_lookup = bundle.get("county_mean_lookup", {})

    # Auto-fill county_mean_coverage from the lookup if the caller passes judet
    # but hasn't explicitly set county_mean_coverage.
    features_dict = dict(features_dict)  # don't mutate caller's dict
    if "county_mean_coverage" not in features_dict and "judet" in features_dict:
        key = _normalize(features_dict["judet"])
        features_dict["county_mean_coverage"] = county_mean_lookup.get(key, 0.0)
        if key not in county_mean_lookup:
            print(f"  Warning: county '{features_dict['judet']}' not found in lookup — county_mean_coverage=0")

    # Build one-row DataFrame with exactly the right columns, fill unknowns with 0
    row = {col: features_dict.get(col, 0) for col in feature_cols}
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

    print(f"\n--- Prediction ---")
    for feat, val in features_dict.items():
        print(f"  {feat}: {val}")
    print(f"  LightGBM:     {pred_lgbm:.1%}")
    print(f"  RandomForest: {pred_rf:.1%}")
    print(f"  XGBoost:      {pred_xgb:.1%}")
    print(f"  Ensemble:     {ensemble:.1%}")

    return {"lgbm": pred_lgbm, "rf": pred_rf, "xgb": pred_xgb, "ensemble": ensemble}


# =========================
# MAIN
# =========================

def main():
    args = _parse_args()
    _apply_config(args)

    print("Loading xlsx locality data...")
    df_xlsx = load_xlsx_data()

    print("\nLoading NUTS3 features from DB...")
    df_nuts3 = load_nuts3_features()

    print("\nLoading building counts from DB...")
    df_buildings = load_building_counts()

    print("\nLoading locality coordinates (geocoded or cached)...")
    df_coords = load_locality_coordinates(df_xlsx)

    print("\nLoading waterway distances (spatial join or cached)...")
    df_coords = load_waterway_features(df_coords)

    print("\nLoading unemployment data (ANOFM PDF or cached)...")
    df_xlsx = load_unemployment_data(df_xlsx)

    print("\nJoining features...")
    df = build_features(df_xlsx, df_nuts3, df_buildings, df_coords)

    print("Preparing model data...")
    X, y, y_orig, df_model = prepare_model_data(df)

    print(f"\nLocalities used: {len(df_model):,}")
    print(df_model[TARGET_COL].describe().round(4))

    print("\nTraining models (5-fold CV)...\n")
    lgbm, rf, xgb, cv_preds, cv_scores = train_models(X, y, y_orig)

    print("\nSaving model bundle...")
    county_mean_lookup = df_model.groupby("judet_key")["coverage_rate"].mean().to_dict()
    save_models(lgbm, rf, xgb, cv_scores, list(X.columns), county_mean_lookup)

    # Save locality feature lookup for predict_location.py (lat/lon → nearest locality)
    lookup_cols = ["localitate", "judet", "judet_key", "n_locuinte", "coverage_rate"] + list(X.columns)
    locality_lookup_path = OUTPUT_DIR / "locality_features.csv"
    df_model[[c for c in lookup_cols if c in df_model.columns]].to_csv(locality_lookup_path, index=False)
    print(f"  Locality lookup saved → {locality_lookup_path}")

    if BOOTSTRAP_ITERATIONS > 0:
        print(f"\nBootstrap prediction intervals (90%, {BOOTSTRAP_ITERATIONS} iterations)...")
        lower, upper = bootstrap_prediction_interval(lgbm, X, y)
        df_model["pred_lower"] = lower
        df_model["pred_upper"] = upper
    else:
        print("\nBootstrap skipped (--no-bootstrap).")

    if not args.no_plots:
        print("\nCorrelation analysis...")
        plot_correlations(X, y_orig)

        print("\nCoverage by settlement type...")
        plot_coverage_by_type(df_model)

        print("\nFeature importance...")
        plot_feature_importance(lgbm, rf, list(X.columns))

        print("\nSHAP analysis...")
        shap_analysis(lgbm, X)

    print("\nPremium potential analysis...")
    df_with_preds = compute_premium_potential(df_model, cv_preds, cv_scores)

    if not args.no_plots:
        print(f"\nCounty focus: {FOCUS_COUNTY}")
        plot_county_focus(df_with_preds, FOCUS_COUNTY)

    print("\nExample standalone prediction (load model from file)...")
    predict_from_features({
        "tip_code":           1,        # 1=Municipiu
        "log_n_locuinte":     np.log1p(85_000),
        "loc_lat":             46.1685,    # Cluj-Napoca approx
        "loc_lon":            21.2882,
        "dist_to_waterway_km": 2.1,
    })


if __name__ == "__main__":
    main()

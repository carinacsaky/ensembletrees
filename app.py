"""
Streamlit app for PAD insurance coverage prediction.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR       = Path(__file__).parent
MODEL_PATH     = BASE_DIR / "output" / "model.pkl"
LOCALITY_TABLE = BASE_DIR / "output" / "locality_features.csv"
OUT            = BASE_DIR / "output"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Run python ensembletrees.py first.")
        sys.exit(1)
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_localities():
    if not LOCALITY_TABLE.exists():
        st.error(f"Locality table not found. Run python ensembletrees.py first.")
        sys.exit(1)
    return pd.read_csv(LOCALITY_TABLE)


def predict(lat, lon):
    bundle = load_bundle()
    df = load_localities()

    lat_col = "loc_lat" if "loc_lat" in df.columns else "lat"
    lon_col = "loc_lon" if "loc_lon" in df.columns else "lon"

    dist_km     = haversine_km(lat, lon, df[lat_col].values, df[lon_col].values)
    nearest_idx = int(np.argmin(dist_km))
    row         = df.iloc[nearest_idx]

    feature_cols = bundle["feature_cols"]
    X_pred = pd.DataFrame([{col: row.get(col, 0) for col in feature_cols}])

    lgbm = bundle["lgbm"]
    rf   = bundle["rf"]
    xgb  = bundle["xgb"]
    cv   = bundle["cv_scores"]

    total = sum(cv.values()) if cv else 0
    if total > 0:
        w = {k: v / total for k, v in cv.items()}
    else:
        w = {"LightGBM": 1/3, "RandomForest": 1/3, "XGBoost": 1/3}

    p_lgbm = sigmoid(lgbm.predict(X_pred)[0])
    p_rf   = sigmoid(rf.predict(X_pred)[0])
    p_xgb  = sigmoid(xgb.predict(X_pred)[0])
    pred   = p_lgbm * w["LightGBM"] + p_rf * w["RandomForest"] + p_xgb * w["XGBoost"]

    avg_flood = row.get("avg_insured_net_flood", 0)
    avg_flood = 0.0 if pd.isna(avg_flood) else float(avg_flood)
    avg_fire  = row.get("avg_insured_net_fire", 0)
    avg_fire  = 0.0 if pd.isna(avg_fire) else float(avg_fire)
    avg_eq    = avg_flood

    dist_ww = float(row.get("dist_to_waterway_km", 2.0) or 2.0)
    if   dist_ww < 0.5: flood_mult = 1.50
    elif dist_ww < 1.5: flood_mult = 1.20
    elif dist_ww < 5.0: flood_mult = 1.00
    else:               flood_mult = 0.80

    dist_vrancea = haversine_km(lat, lon, 45.7, 26.6)
    if   dist_vrancea < 100:  eq_mult = 1.50
    elif dist_vrancea < 200:  eq_mult = 1.30
    elif dist_vrancea < 350:  eq_mult = 1.10
    else:                     eq_mult = 0.80

    hazard      = float(row.get("hazard_intensity", 1.0) or 1.0)
    hazard_mult = max(0.5, min(hazard, 2.0))

    prem_flood_lo = avg_flood * 0.0005 * flood_mult * hazard_mult
    prem_flood_hi = avg_flood * 0.0020 * flood_mult * hazard_mult
    prem_eq_lo    = avg_eq   * 0.0003 * eq_mult
    prem_eq_hi    = avg_eq   * 0.0010 * eq_mult
    prem_fire_lo  = avg_fire * 0.0003
    prem_fire_hi  = avg_fire * 0.0008

    return {
        "locality":      row["localitate"],
        "county":        row["judet"],
        "dist_km":       float(dist_km[nearest_idx]),
        "pred":          pred,
        "act_rate":      row.get("coverage_rate"),
        "n_locuinte":    row.get("n_locuinte", 0),
        "dist_ww":       dist_ww,
        "flood_mult":    flood_mult,
        "dist_vrancea":  dist_vrancea,
        "eq_mult":       eq_mult,
        "avg_flood":     avg_flood,
        "avg_fire":      avg_fire,
        "prem_flood_lo": prem_flood_lo,
        "prem_flood_hi": prem_flood_hi,
        "prem_eq_lo":    prem_eq_lo,
        "prem_eq_hi":    prem_eq_hi,
        "prem_fire_lo":  prem_fire_lo,
        "prem_fire_hi":  prem_fire_hi,
    }


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Insurance Predictor", page_icon="🏠", layout="wide")
st.title("Insurance Coverage Predictor")
st.caption("Romania — LightGBM + RandomForest + XGBoost ensemble")

tab_predict, tab_model, tab_shap, tab_market, tab_county, tab_history = st.tabs([
    "Predict", "Model Analysis", "SHAP", "Market Overview", "County Focus", "Prediction History"
])

# ── Tab 1: Predict ────────────────────────────────────────────────────────────
with tab_predict:
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        lat = col1.number_input("Latitude",  min_value=43.0, max_value=49.0, value=47.16, step=0.01, format="%.4f")
        lon = col2.number_input("Longitude", min_value=20.0, max_value=30.0, value=27.59, step=0.01, format="%.4f")
        submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

    if submitted:
        with st.spinner("Running prediction..."):
            r = predict(lat, lon)

        pred      = r["pred"]
        n_loc     = r["n_locuinte"] or 0
        act_rate  = r["act_rate"]
        est       = int(round(n_loc * pred)) if n_loc > 0 else 0
        gap       = int(round(n_loc * max(pred - float(act_rate or 0), 0))) if n_loc > 0 else 0

        if   pred >= 0.35: tier = "HIGH"
        elif pred >= 0.20: tier = "MEDIUM"
        else:              tier = "LOW"

        st.subheader(f"{r['locality']}, {r['county']}")
        st.caption(f"{r['dist_km']:.1f} km from query point · {lat:.4f}°N, {lon:.4f}°E")

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted coverage", f"{pred:.1%}",
                  delta=f"{(pred - float(act_rate)):.1%} vs actual" if act_rate else None)
        c2.metric("Tier", tier)
        c3.metric("Actual coverage", f"{float(act_rate):.1%}" if act_rate else "—")

        if n_loc > 0:
            c4, c5, c6 = st.columns(3)
            c4.metric("Housing units",      f"{int(n_loc):,}")
            c5.metric("Est. PAD policies",  f"{est:,}")
            c6.metric("Opportunity gap/yr", f"{gap * 20:,.0f} EUR")

        if r["avg_flood"] > 0:
            st.divider()
            st.subheader("Private Insurance Premium Estimate")
            p1, p2, p3 = st.columns(3)
            p1.metric("Flood premium/yr",      f"{r['prem_flood_lo']:,.0f} – {r['prem_flood_hi']:,.0f} EUR",
                      delta=f"×{r['flood_mult']:.2f} flood risk ({r['dist_ww']:.1f} km to waterway)", delta_color="off")
            p2.metric("Earthquake premium/yr", f"{r['prem_eq_lo']:,.0f} – {r['prem_eq_hi']:,.0f} EUR",
                      delta=f"×{r['eq_mult']:.2f} eq risk ({r['dist_vrancea']:.0f} km to Vrancea)", delta_color="off")
            p3.metric("Fire premium/yr",       f"{r['prem_fire_lo']:,.0f} – {r['prem_fire_hi']:,.0f} EUR")

            total_lo = r["prem_flood_lo"] + r["prem_eq_lo"] + r["prem_fire_lo"]
            total_hi = r["prem_flood_hi"] + r["prem_eq_hi"] + r["prem_fire_hi"]
            st.info(f"**Total estimated private premium: {total_lo:,.0f} – {total_hi:,.0f} EUR/yr**  "
                    f"(avg insured value: {r['avg_flood']:,.0f} EUR)")

# ── Tab 2: Model Analysis ─────────────────────────────────────────────────────
with tab_model:
    bundle = load_bundle()
    cv = bundle.get("cv_scores", {})

    if cv:
        st.subheader("Cross-validation R² scores")
        cols = st.columns(len(cv))
        for col, (name, score) in zip(cols, cv.items()):
            col.metric(name, f"{score:.3f}")

    st.divider()

    img_feat = OUT / "feature_importance.png"
    img_corr = OUT / "correlations.png"
    img_cov  = OUT / "coverage_by_type.png"

    if img_feat.exists():
        st.subheader("Feature Importance")
        st.image(str(img_feat), use_container_width=True)

    if img_corr.exists():
        st.subheader("Correlations")
        st.image(str(img_corr), use_container_width=True)

    if img_cov.exists():
        st.subheader("Coverage by Locality Type")
        st.image(str(img_cov), use_container_width=True)

# ── Tab 3: SHAP ───────────────────────────────────────────────────────────────
with tab_shap:
    shap_summary = OUT / "shap_summary.png"
    if shap_summary.exists():
        st.subheader("SHAP Summary")
        st.image(str(shap_summary), use_container_width=True)

    st.subheader("SHAP Dependence Plots")
    shap_plots = sorted(OUT.glob("shap_dep_*.png"))
    if shap_plots:
        cols = st.columns(2)
        for i, p in enumerate(shap_plots):
            feature_name = p.stem.replace("shap_dep_", "").replace("_", " ")
            cols[i % 2].image(str(p), caption=feature_name, use_container_width=True)
    else:
        st.info("No SHAP dependence plots found.")

# ── Tab 4: Market Overview ────────────────────────────────────────────────────
with tab_market:
    img_prem = OUT / "premium_potential.png"
    if img_prem.exists():
        st.subheader("Premium Potential by County")
        st.image(str(img_prem), use_container_width=True)

    csv_prem = OUT / "premium_potential.csv"
    if csv_prem.exists():
        st.subheader("Premium Potential Table")
        df_prem = pd.read_csv(csv_prem)
        st.dataframe(df_prem, use_container_width=True, hide_index=True)

    csv_county = OUT / "county_summary.csv"
    if csv_county.exists():
        st.subheader("County Summary")
        df_county = pd.read_csv(csv_county)
        st.dataframe(df_county, use_container_width=True, hide_index=True)

    csv_cross = OUT / "cross_analysis.csv"
    if csv_cross.exists():
        with st.expander("Cross Analysis"):
            df_cross = pd.read_csv(csv_cross)
            st.dataframe(df_cross, use_container_width=True, hide_index=True)

    csv_priv = OUT / "private_penetration.csv"
    if csv_priv.exists():
        with st.expander("Private Insurance Penetration"):
            df_priv = pd.read_csv(csv_priv)
            st.dataframe(df_priv, use_container_width=True, hide_index=True)

# ── Tab 5: County Focus ───────────────────────────────────────────────────────
with tab_county:
    county_imgs = sorted(OUT.glob("county_focus_*.png"))
    if county_imgs:
        for img in county_imgs:
            county_name = img.stem.replace("county_focus_", "").replace("_", " ").title()
            st.subheader(county_name)
            st.image(str(img), use_container_width=True)
    else:
        st.info("No county focus charts found.")

# ── Tab 6: Prediction History ─────────────────────────────────────────────────
with tab_history:
    csv_log = OUT / "predictions_log.csv"
    if csv_log.exists():
        df_log = pd.read_csv(csv_log)
        st.subheader(f"Prediction Log ({len(df_log)} runs)")
        st.dataframe(df_log, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df_log.to_csv(index=False),
                           file_name="predictions_log.csv", mime="text/csv")
    else:
        st.info("No predictions yet. Use the Predict tab to run your first prediction.")

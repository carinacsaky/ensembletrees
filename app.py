"""
Streamlit app for PAD insurance coverage prediction.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

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
    w = {k: v / total for k, v in cv.items()} if total > 0 else {"LightGBM": 1/3, "RandomForest": 1/3, "XGBoost": 1/3}

    p_lgbm = sigmoid(lgbm.predict(X_pred)[0])
    p_rf   = sigmoid(rf.predict(X_pred)[0])
    p_xgb  = sigmoid(xgb.predict(X_pred)[0])
    pred   = p_lgbm * w["LightGBM"] + p_rf * w["RandomForest"] + p_xgb * w["XGBoost"]

    avg_flood = 0.0 if pd.isna(row.get("avg_insured_net_flood", 0)) else float(row.get("avg_insured_net_flood", 0))
    avg_fire  = 0.0 if pd.isna(row.get("avg_insured_net_fire",  0)) else float(row.get("avg_insured_net_fire",  0))

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

    hazard_mult = max(0.5, min(float(row.get("hazard_intensity", 1.0) or 1.0), 2.0))

    prem_flood_lo = avg_flood * 0.0005 * flood_mult * hazard_mult
    prem_flood_hi = avg_flood * 0.0020 * flood_mult * hazard_mult
    prem_eq_lo    = avg_flood * 0.0003 * eq_mult
    prem_eq_hi    = avg_flood * 0.0010 * eq_mult
    prem_fire_lo  = avg_fire  * 0.0003
    prem_fire_hi  = avg_fire  * 0.0008

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


@st.cache_data(show_spinner=False)
def build_map(counties_filter, tiers_filter, pred_lat, pred_lon,
              pred_locality=None, pred_county=None, pred_coverage=None,
              pred_total_lo=None, pred_total_hi=None):
    df = load_localities()
    lat_col = "loc_lat" if "loc_lat" in df.columns else "lat"
    lon_col = "loc_lon" if "loc_lon" in df.columns else "lon"

    if counties_filter:
        df = df[df["judet"].isin(counties_filter)]

    def tier_of(rate):
        if pd.isna(rate): return None
        return "HIGH" if rate >= 0.35 else ("MEDIUM" if rate >= 0.20 else "LOW")

    df = df[df["coverage_rate"].apply(tier_of).isin(tiers_filter)]

    max_loc = df["n_locuinte"].replace(0, np.nan).max() or 1
    def dot_radius(n):
        if pd.isna(n) or n <= 0: return 4
        return float(np.clip(3 + 11 * np.log1p(n) / np.log1p(max_loc), 3, 14))

    center = [pred_lat, pred_lon] if pred_lat else [45.9, 24.9]
    zoom   = 10 if pred_lat else 7
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    for _, row in df.iterrows():
        rate = row.get("coverage_rate")
        if pd.isna(row[lat_col]) or pd.isna(row[lon_col]):
            continue

        if pd.isna(rate):
            color = "#aaaaaa"
        else:
            r_val = float(rate)
            if r_val <= 0.20:
                t = r_val / 0.20
                rc, gc = int(220), int(t * 200)
            else:
                t = min((r_val - 0.20) / 0.30, 1.0)
                rc, gc = int(220 - t * 200), int(200)
            color = f"#{rc:02x}{gc:02x}00"

        n_loc     = row.get("n_locuinte", 0) or 0
        avg_flood = 0.0 if pd.isna(row.get("avg_insured_net_flood", 0)) else float(row.get("avg_insured_net_flood", 0))
        avg_fire  = 0.0 if pd.isna(row.get("avg_insured_net_fire",  0)) else float(row.get("avg_insured_net_fire",  0))

        popup_lines = [f"<b>{row['localitate']}, {row['judet']}</b>"]
        if not pd.isna(rate):
            t2 = tier_of(rate)
            popup_lines.append(f"Coverage: {float(rate):.1%} [{t2}]")
        if n_loc > 0:
            popup_lines.append(f"Housing units: {int(n_loc):,}")
            popup_lines.append(f"Est. PAD policies: {int(n_loc * float(rate or 0)):,}")
            popup_lines.append(f"Opportunity gap: {int(n_loc * max(0.35 - float(rate or 0), 0)) * 20:,.0f} EUR/yr")
        if avg_flood > 0:
            dist_ww = float(row.get("dist_to_waterway_km", 2.0) or 2.0)
            if   dist_ww < 0.5: fm = 1.50
            elif dist_ww < 1.5: fm = 1.20
            elif dist_ww < 5.0: fm = 1.00
            else:               fm = 0.80
            dv = haversine_km(float(row[lat_col]), float(row[lon_col]), 45.7, 26.6)
            em = 1.50 if dv < 100 else (1.30 if dv < 200 else (1.10 if dv < 350 else 0.80))
            hm = max(0.5, min(float(row.get("hazard_intensity", 1.0) or 1.0), 2.0))
            plo = avg_flood * 0.0005 * fm * hm + avg_flood * 0.0003 * em + avg_fire * 0.0003
            phi = avg_flood * 0.0020 * fm * hm + avg_flood * 0.0010 * em + avg_fire * 0.0008
            popup_lines += [f"<br><b>Private premium est.:</b>",
                            f"Total: {plo:,.0f} – {phi:,.0f} EUR/yr"]

        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=dot_radius(row.get("n_locuinte")),
            color=color, fill=True, fill_color=color, fill_opacity=0.85,
            popup=folium.Popup("<br>".join(popup_lines), max_width=250),
            tooltip=f"{row['localitate']} — {float(rate):.1%}" if not pd.isna(rate) else row['localitate'],
        ).add_to(m)

    # Blue pin for query point
    if pred_lat:
        lines = [f"<b>{pred_locality}, {pred_county}</b>"]
        if pred_coverage is not None:
            lines.append(f"Predicted coverage: {pred_coverage:.1%}")
        if pred_total_lo is not None:
            lines.append(f"Private premium: {pred_total_lo:,.0f} – {pred_total_hi:,.0f} EUR/yr")
        folium.Marker(
            location=[pred_lat, pred_lon],
            popup=folium.Popup("<br>".join(lines), max_width=220),
            tooltip=f"{pred_locality} — {pred_coverage:.1%}" if pred_coverage else "Query point",
            icon=folium.Icon(color="blue", icon="map-marker"),
        ).add_to(m)

    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px 14px; border-radius:8px;
                border:1px solid #ccc; font-size:13px; line-height:1.7;">
        <b>PAD Coverage Rate</b><br>
        <span style="background:linear-gradient(to right,#dc0000,#dcc800,#00c800);
                     display:inline-block;width:120px;height:10px;border-radius:4px;
                     vertical-align:middle;"></span><br>
        0% ← low &nbsp;&nbsp; high → 50%+<br>
        <span style="color:#aaa">●</span> No data &nbsp;
        <span style="font-size:11px;">dot size = population</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Insurance Predictor", page_icon="🏠", layout="wide")

st.markdown("""
<style>
div.stButton > button[kind="primary"],
div.stFormSubmitButton > button[kind="primaryFormSubmit"] {
    background-color: #1a6fba;
    border-color: #1a6fba;
    color: white;
}
div.stButton > button[kind="primary"]:hover,
div.stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {
    background-color: #155a99;
    border-color: #155a99;
}
</style>
""", unsafe_allow_html=True)

st.title("Insurance Coverage Predictor")
st.caption("Romania — LightGBM + RandomForest + XGBoost ensemble")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search by Locality")
    df_all = load_localities()
    lat_col_all = "loc_lat" if "loc_lat" in df_all.columns else "lat"
    lon_col_all = "loc_lon" if "loc_lon" in df_all.columns else "lon"

    locality_names = sorted(df_all["localitate"].dropna().unique().tolist())
    selected_locality = st.selectbox("Type a locality name", [""] + locality_names)

    if selected_locality:
        loc_row = df_all[df_all["localitate"] == selected_locality].iloc[0]
        st.session_state["search_lat"] = float(loc_row[lat_col_all])
        st.session_state["search_lon"] = float(loc_row[lon_col_all])
        st.success(f"{selected_locality}, {loc_row['judet']}")
        st.caption(f"{loc_row[lat_col_all]:.4f}°N, {loc_row[lon_col_all]:.4f}°E")

    st.divider()
    st.header("Map Filters")
    counties = sorted(df_all["judet"].dropna().unique().tolist())
    selected_counties = st.multiselect("Filter by county", counties, default=[])

selected_tiers = ["HIGH", "MEDIUM", "LOW"]

tab_predict, tab_market, tab_history = st.tabs([
    "Predict", "Market Overview", "Prediction History"
])

# ── Tab 1: Predict ────────────────────────────────────────────────────────────
with tab_predict:
    default_lat = st.session_state.pop("search_lat", 47.16)
    default_lon = st.session_state.pop("search_lon", 27.59)

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        lat = col1.number_input("Latitude",  min_value=43.0, max_value=49.0, value=float(default_lat), step=0.01, format="%.4f")
        lon = col2.number_input("Longitude", min_value=20.0, max_value=30.0, value=float(default_lon), step=0.01, format="%.4f")
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        with st.spinner("Running prediction..."):
            r = predict(lat, lon)
        st.session_state["last_prediction"] = {"lat": lat, "lon": lon, "result": r}

        pred     = r["pred"]
        n_loc    = r["n_locuinte"] or 0
        act_rate = r["act_rate"]
        est      = int(round(n_loc * pred)) if n_loc > 0 else 0
        gap      = int(round(n_loc * max(pred - float(act_rate or 0), 0))) if n_loc > 0 else 0

        if   pred >= 0.35: tier, tier_color = "HIGH",   "#1a6fba"
        elif pred >= 0.20: tier, tier_color = "MEDIUM", "#e07b00"
        else:              tier, tier_color = "LOW",    "#cc2200"

        st.subheader(f"{r['locality']}, {r['county']}")
        st.caption(f"{r['dist_km']:.1f} km from query point · {lat:.4f}°N, {lon:.4f}°E")

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted coverage", f"{pred:.1%}",
                  delta=f"{(pred - float(act_rate)):.1%} vs actual" if act_rate else None)
        c2.markdown(f"**Tier**<br><span style='color:{tier_color};font-size:1.4rem;font-weight:700'>{tier}</span>", unsafe_allow_html=True)
        c3.metric("Actual coverage", f"{float(act_rate):.1%}" if act_rate else "—")

        nat_avg = float(load_localities()["coverage_rate"].mean())
        st.markdown(f"**Coverage vs national average ({nat_avg:.1%})**")
        st.progress(min(pred, 1.0), text=f"Predicted: {pred:.1%}")
        if act_rate:
            st.progress(min(float(act_rate), 1.0), text=f"Actual: {float(act_rate):.1%}")
        st.progress(min(nat_avg, 1.0), text=f"National avg: {nat_avg:.1%}")

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

    # ── Map ───────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Map")
    st.caption("Click any dot to see details. Blue pin = your query point.")

    pred_state = st.session_state.get("last_prediction")
    pred_lat   = pred_state["lat"] if pred_state else None
    pred_lon   = pred_state["lon"] if pred_state else None

    if pred_state:
        rr       = pred_state["result"]
        total_lo = rr["prem_flood_lo"] + rr["prem_eq_lo"] + rr["prem_fire_lo"]
        total_hi = rr["prem_flood_hi"] + rr["prem_eq_hi"] + rr["prem_fire_hi"]
        fmap = build_map(
            tuple(selected_counties), tuple(selected_tiers),
            pred_lat, pred_lon,
            rr["locality"], rr["county"], rr["pred"],
            total_lo if rr["avg_flood"] > 0 else None,
            total_hi if rr["avg_flood"] > 0 else None,
        )
        st.info(f"Last prediction: **{rr['locality']}, {rr['county']}** — {rr['pred']:.1%} coverage")
    else:
        fmap = build_map(tuple(selected_counties), tuple(selected_tiers), None, None)

    st_folium(fmap, use_container_width=True, height=600)

    # ── County Summary Table ──────────────────────────────────────────────────
    st.divider()
    st.subheader("County Summary")

    @st.cache_data(show_spinner=False)
    def county_summary_table(counties_filter):
        df = load_localities()
        if counties_filter:
            df = df[df["judet"].isin(counties_filter)]
        grp = df.groupby("judet").agg(
            localities       = ("localitate",    "count"),
            avg_coverage     = ("coverage_rate",  "mean"),
            min_coverage     = ("coverage_rate",  "min"),
            max_coverage     = ("coverage_rate",  "max"),
            total_housing    = ("n_locuinte",      "sum"),
        ).reset_index()
        grp["opportunity_eur"] = (
            grp["total_housing"] * (0.35 - grp["avg_coverage"].clip(upper=0.35)) * 20
        ).round(0).astype(int)
        grp["avg_coverage"] = grp["avg_coverage"].map(lambda x: f"{x:.1%}")
        grp["min_coverage"] = grp["min_coverage"].map(lambda x: f"{x:.1%}")
        grp["max_coverage"] = grp["max_coverage"].map(lambda x: f"{x:.1%}")
        grp["total_housing"] = grp["total_housing"].map(lambda x: f"{int(x):,}")
        grp["opportunity_eur"] = grp["opportunity_eur"].map(lambda x: f"{x:,}")
        grp.columns = ["County", "Localities", "Avg Coverage", "Min Coverage",
                       "Max Coverage", "Total Housing", "Opportunity Gap (EUR/yr)"]
        return grp.sort_values("Avg Coverage")

    df_summary = county_summary_table(tuple(selected_counties))
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

# ── Tab 2: Market Overview ────────────────────────────────────────────────────
with tab_market:
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

# ── Tab 3: Prediction History ─────────────────────────────────────────────────
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

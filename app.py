# app.py — SafeHer with added features (fixed language scoping bug, white background, pie chart, added states)
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import time
import base64
import pathlib
from datetime import datetime

# Optional: clustering (import inside try so app works if sklearn missing)
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="SafeHer — Women Safety", layout="wide", initial_sidebar_state="expanded")

# -----------------------
# Raw image URL (if you deploy) - optional
# -----------------------
RAW_IMAGE_URL = ""  # set to your raw.githubusercontent link if deploying

# -----------------------
# Simple sample data (includes Month & Hour for trend charts)
# -----------------------
SAMPLE = {
    "Telangana": [
        {"City":"Hyderabad","Victims":120,"Latitude":17.3850,"Longitude":78.4867},
        {"City":"Warangal","Victims":25,"Latitude":17.9789,"Longitude":79.5941},
        {"City":"Nizamabad","Victims":12,"Latitude":18.6722,"Longitude":78.0941},
        {"City":"Karimnagar","Victims":18,"Latitude":18.4386,"Longitude":79.1288},
    ],
    "Maharashtra": [
        {"City":"Mumbai","Victims":200,"Latitude":19.0760,"Longitude":72.8777},
        {"City":"Pune","Victims":80,"Latitude":18.5204,"Longitude":73.8567},
        {"City":"Nagpur","Victims":30,"Latitude":21.1458,"Longitude":79.0882},
        {"City":"Nashik","Victims":20,"Latitude":19.9975,"Longitude":73.7898},
    ],
    "Karnataka": [
        {"City":"Bengaluru","Victims":140,"Latitude":12.9716,"Longitude":77.5946},
        {"City":"Mysuru","Victims":22,"Latitude":12.2958,"Longitude":76.6394},
        {"City":"Hubli","Victims":10,"Latitude":15.3647,"Longitude":75.1235},
    ],
    "Delhi": [
        {"City":"Central Delhi","Victims":160,"Latitude":28.6325,"Longitude":77.2195},
        {"City":"South Delhi","Victims":90,"Latitude":28.5245,"Longitude":77.1855},
        {"City":"North Delhi","Victims":70,"Latitude":28.7041,"Longitude":77.1025},
    ],
    # --- ADDED STATES BELOW ---
    "West Bengal": [
        {"City":"Kolkata","Victims":150,"Latitude":22.5726,"Longitude":88.3639},
        {"City":"Howrah","Victims":45,"Latitude":22.5958,"Longitude":88.2636},
        {"City":"Durgapur","Victims":15,"Latitude":23.5350,"Longitude":87.3175},
    ],
    "Gujarat": [
        {"City":"Ahmedabad","Victims":110,"Latitude":23.0225,"Longitude":72.5714},
        {"City":"Surat","Victims":65,"Latitude":21.1702,"Longitude":72.8311},
        {"City":"Vadodara","Victims":35,"Latitude":22.3072,"Longitude":73.1812},
    ],
    "Tamil Nadu": [
        {"City":"Chennai","Victims":130,"Latitude":13.0827,"Longitude":80.2707},
        {"City":"Coimbatore","Victims":40,"Latitude":11.0168,"Longitude":76.9558},
        {"City":"Madurai","Victims":25,"Latitude":9.9252,"Longitude":78.1198},
    ],
    # --- ADDED STATES ABOVE ---
}

# -----------------------
# Helpers
# -----------------------
def sample_to_df(state):
    rows = SAMPLE.get(state, [])
    df = pd.DataFrame(rows)
    # generate Month and Hour for demo charts if not present
    if "Month" not in df.columns:
        months = np.random.choice(
            ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
            size=len(df)
        )
        df["Month"] = months
    if "Hour" not in df.columns:
        df["Hour"] = np.random.choice(list(range(0,24)), size=len(df))
    return df

def compute_risk_score(df):
    if df.empty:
        return df
    # If sklearn not available, use a simple min-max without importing inside to avoid errors
    try:
        from sklearn.preprocessing import MinMaxScaler
        sc = MinMaxScaler()
        df = df.copy()
        if "Victims" not in df.columns:
            df["Victims"] = 0
        df["Victims_norm"] = sc.fit_transform(df[["Victims"]])
    except Exception:
        df = df.copy()
        if "Victims" not in df.columns:
            df["Victims"] = 0
        v = df["Victims"].astype(float).fillna(0).values
        if len(v) > 0 and v.max() != v.min():
            df["Victims_norm"] = (v - v.min()) / (v.max() - v.min())
        else:
            df["Victims_norm"] = 0.0

    np.random.seed(0)
    df["recency_factor"] = np.random.rand(len(df)) * 0.2
    df["risk_score"] = 0.8 * df["Victims_norm"] + 0.2 * df["recency_factor"]
    df["Risk"] = pd.cut(df["risk_score"], bins=[-1,0.33,0.66,1.0], labels=["Low","Medium","High"])
    return df

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dLon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# background helper (set to white background)
def set_landing_background():
    # Force white background on the entire app and remove landing-overlay styling
    css = """
    <style>
    /* Force white background */
    .stApp {
        background-color: white; 
        background-image: none;
    }
    /* Remove text overlay background/color for main landing content against white */
    .landing-overlay {
        background: none; /* Remove dark overlay */
        padding: 26px;
        border-radius: 12px;
        color: black; /* Change text color to black for white background */
        max-width: 1000px;
        margin: 40px auto;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -----------------------
# Multi-language simple dictionary
# -----------------------
LANGS = {
    "en": {
        "enter": "Enter SafeHer",
        "select_state": "Select State",
        "upload_csv": "Upload CSV (optional)",
        "panic": "PANIC BUTTON",
        "panic_confirm": "Emergency alert (UI only) activated!",
        "summary": "Summary",
        "trends": "Time-Based Trends",
        "interactive_table": "Interactive Table",
        "map": "Interactive Map",
        "proximity": "Proximity Check",
        "back_home": "⟵ Back to Home",
        "no_data": "No data for selected state. Upload CSV with required columns or choose another state."
    },
    "hi": {
        "enter": "SafeHer में प्रवेश करें",
        "select_state": "राज्य चुनें",
        "upload_csv": "CSV अपलोड करें (वैकल्पिक)",
        "panic": "आपात बटन",
        "panic_confirm": "आपातकालीन चेतावनी (केवल UI) सक्रिय!",
        "summary": "सारांश",
        "trends": "समय-आधारित प्रवृत्तियाँ",
        "interactive_table": "इंटरएक्टिव तालिका",
        "map": "इंटरएक्टिव मानचित्र",
        "proximity": "निकटता जाँच",
        "back_home": "⟵ घर पर वापस",
        "no_data": "चयनित राज्य के लिए डेटा नहीं। आवश्यक कॉलम के साथ CSV अपलोड करें या दूसरा राज्य चुनें।"
    },
    "te": {
        "enter": "SafeHer లో ప్రవేశించండి",
        "select_state": "రాష్ట్రం ఎంచుకోండి",
        "upload_csv": "CSV అప్లోడ్ చేయండి (ఐచ్ఛికం)",
        "panic": "పేనిక్ బటన్",
        "panic_confirm": "విపత్తు హెచ్చరిక (UI మాత్రమే) ప్రారంభమైంది!",
        "summary": "సారాంశం",
        "trends": "సమయానికి సంబంధించిన ధోరణులు",
        "interactive_table": "ఇంటరాక్టివ్ పట్టిక",
        "map": "ఇంటరాక్టివ్ మ్యాప్",
        "proximity": "సమీపతా తనిఖీ",
        "back_home": "⟵ హోమ్‌కు వెళ్ళు",
        "no_data": "ఎంపిక చేసిన రాష్ట్రానికి డేటా లేదు. అవసరమైన కాలమ్స్‌తో CSVను అప్లోడ్ చేయండి లేదా వేరే రాష్ట్రాన్ని ఎంచుకోండి."
    }
}

# -----------------------
# Session state for page & language
# -----------------------
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# -----------------------
# Landing page
# -----------------------
def landing_page():
    # CUSTOM CSS FOR BUTTON VISIBILITY
    st.markdown("""
    <style>
    /* Targeting the primary button on the landing page for high contrast */
    .landing-overlay .stButton>button {
        background-color: #e91e63; /* Bright Magenta */
        color: white; /* White text */
        border: none;
        padding: 10px 24px;
        font-size: 20px;
        font-weight: bold;
        border-radius: 8px;
        transition: background-color 0.3s;
    }
    .landing-overlay .stButton>button:hover {
        background-color: #c2185b; /* Darker magenta on hover */
    }
    </style>
    """, unsafe_allow_html=True)
    
    # use current session language
    lang_code = st.session_state.get("lang", "en")
    L = LANGS.get(lang_code, LANGS["en"])

    set_landing_background() # Sets global background to white
    st.markdown("<div class='landing-overlay'>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size:44px; margin-bottom:4px;'>🛡️ SafeHer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:18px;'>Women Safety Prediction & Alert System — data-driven risk maps & alerts</p>", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button(L["enter"], use_container_width=True):
            st.session_state.page = "main"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Main page + features
# -----------------------
def main_page():
    # Determine language and labels locally (do NOT reassign global L)
    lang_code = st.session_state.get("lang", "en")
    local_L = LANGS.get(lang_code, LANGS["en"])
    
    # Ensure white background is set for main page too
    set_landing_background()

    # top controls
    top_left, top_center, top_right = st.columns([1,2,1])
    with top_left:
        if st.button(local_L["back_home"]):
            st.session_state.page = "landing"
            st.rerun()
    with top_center:
        st.title("🛡️ SafeHer — Women Safety Dashboard")
    with top_right:
        # language selector; map labels to codes
        lang_map = {"English":"en", "हिन्दी":"hi", "తెలుగు":"te"}
        reverse_map = {v:k for k,v in lang_map.items()}
        current_label = reverse_map.get(lang_code, "English")
        chosen_label = st.selectbox("Language", options=list(lang_map.keys()), index=list(lang_map.keys()).index(current_label))
        chosen_code = lang_map[chosen_label]
        if chosen_code != lang_code:
            st.session_state["lang"] = chosen_code
            st.rerun()

    # Use local_L from now on
    L = local_L

    st.sidebar.title("Settings")
    uploaded = st.sidebar.file_uploader(L["upload_csv"], type=["csv"])
    st.sidebar.markdown("---")
    st.sidebar.write("Feature toggles")
    show_heatmap = st.sidebar.checkbox("Show Heatmap", value=False)
    show_table = st.sidebar.checkbox("Show Interactive Table", value=True)

    # Panic button UI (no SMS)
    st.markdown("### 🔴 " + L["panic"])
    if st.button("🔺 " + L["panic"]):
        st.error(L["panic_confirm"])
        st.balloons()

    # Load data
    if uploaded is not None:
        try:
            df_all = pd.read_csv(uploaded)
            st.sidebar.success("CSV loaded")
            df_all.columns = [c.strip() for c in df_all.columns]
        except Exception:
            st.sidebar.error("Could not load CSV — using sample")
            df_all = None
    else:
        df_all = None

    # states list
    states = sorted(SAMPLE.keys()) if df_all is None or "State" not in df_all.columns else sorted(df_all["State"].unique().tolist())
    state = st.selectbox(L["select_state"], options=states)

    # build state_df
    if df_all is not None and "State" in df_all.columns:
        state_df = df_all[df_all["State"] == state].copy()
    elif df_all is not None and "City" in df_all.columns:
        sample_cities = [r["City"] for r in SAMPLE.get(state,[])]
        state_df = df_all[df_all["City"].isin(sample_cities)].copy()
    else:
        state_df = sample_to_df(state)

    if state_df is None or state_df.empty:
        st.warning(L["no_data"])
        return

    # ensure required cols and fill lat/lon if missing using sample
    if "Victims" not in state_df.columns:
        state_df["Victims"] = 0
    if "Latitude" not in state_df.columns or "Longitude" not in state_df.columns:
        sample_df = sample_to_df(state)
        if not sample_df.empty:
            mapping = sample_df.set_index("City")[["Latitude","Longitude"]].to_dict(orient="index")
            def fill_lat(row):
                return mapping.get(row.get("City"), {}).get("Latitude", np.nan)
            def fill_lon(row):
                return mapping.get(row.get("City"), {}).get("Longitude", np.nan)
            state_df["Latitude"] = state_df.apply(lambda r: fill_lat(r) if pd.isna(r.get("Latitude")) else r.get("Latitude"), axis=1)
            state_df["Longitude"] = state_df.apply(lambda r: fill_lon(r) if pd.isna(r.get("Longitude")) else r.get("Longitude"), axis=1)

    # Create Month/Hour if absent
    if "Month" not in state_df.columns:
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        state_df["Month"] = [months[i % 12] for i in range(len(state_df))]
    if "Hour" not in state_df.columns:
        state_df["Hour"] = [int((i*3) % 24) for i in range(len(state_df))]

    # compute risk scores
    state_df["Victims"] = pd.to_numeric(state_df["Victims"], errors="coerce").fillna(0)
    state_df = compute_risk_score(state_df)

    # --- SUMMARY DASHBOARD ---
    st.header(L["summary"])
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.metric("Total Cities", len(state_df))
    with c2:
        st.metric("Total Victims", int(state_df["Victims"].sum()))
    with c3:
        st.metric("High Risk Cities", int((state_df["Risk"]=="High").sum()))
    with c4:
        st.metric("Median Victims", int(state_df["Victims"].median()))

    top_city = state_df.sort_values("Victims", ascending=False).iloc[0]
    low_city = state_df.sort_values("Victims", ascending=True).iloc[0]
    st.markdown(f"**Top city (most victims):** {top_city['City']} — {int(top_city['Victims'])}")
    st.markdown(f"**Lowest city (least victims):** {low_city['City']} — {int(low_city['Victims'])}")

    # --- TIME-BASED GRAPHS ---
    st.header(L["trends"])
    
    col_m, col_h = st.columns(2)

    # Month trend (line plot)
    with col_m:
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        month_counts = state_df.groupby("Month")["Victims"].sum().reindex(month_order).fillna(0)
        fig_m, ax_m = plt.subplots(figsize=(7,4))
        sns.lineplot(x=month_counts.index, y=month_counts.values, marker="o", ax=ax_m)
        ax_m.set_title("Victims by Month")
        ax_m.set_ylabel("Victims")
        st.pyplot(fig_m)

    # Hour distribution (PIE CHART)
    with col_h:
        hour_counts = state_df.groupby("Hour")["Victims"].sum().reindex(range(0,24), fill_value=0)
        # Filter hours where Victims > 0 for a cleaner pie chart
        pie_data = hour_counts[hour_counts > 0]
        
        # Prepare labels: only show the hour if it has victims
        # Use a maximum of 12 labels for readability; group small hours into 'Other'
        if len(pie_data) > 12:
            # Group all but the top 11 hours into 'Other'
            top_11_hours = pie_data.nlargest(11)
            other_victims = pie_data.sum() - top_11_hours.sum()
            
            pie_values = top_11_hours.values.tolist()
            pie_labels = [f"Hour {h}" for h in top_11_hours.index]
            
            if other_victims > 0:
                pie_values.append(other_victims)
                pie_labels.append("Other Hours")
        else:
            pie_values = pie_data.values.tolist()
            pie_labels = [f"Hour {h}" for h in pie_data.index]
        
        fig_h, ax_h = plt.subplots(figsize=(7,4))
        
        # Pie chart configuration
        ax_h.pie(
            pie_values, 
            labels=pie_labels, 
            autopct='%1.1f%%', 
            startangle=90, 
            pctdistance=0.85 # Position of the percentage label
        )
        ax_h.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        ax_h.set_title("Victims Distribution by Hour of Day")
        st.pyplot(fig_h)

    # --- MAP ---
    st.header(L["map"])
    center_lat = state_df["Latitude"].mean()
    center_lon = state_df["Longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, control_scale=True)

    if show_heatmap:
        try:
            from folium.plugins import HeatMap
            heat_data = state_df[["Latitude","Longitude","Victims"]].dropna().values.tolist()
            HeatMap([[r[0], r[1], r[2]] for r in heat_data], radius=25).add_to(m)
        except Exception:
            pass

    for _, row in state_df.iterrows():
        color = "red" if row["Risk"] == "High" else ("orange" if row["Risk"] == "Medium" else "green")
        folium.CircleMarker(
            [row["Latitude"], row["Longitude"]],
            radius = 6 + min(18, max(4, (row["Victims"]**0.5))),
            color=color, fill=True, fill_color=color, fill_opacity=0.7,
            popup=f"{row['City']} — Victims: {int(row['Victims'])} — Risk: {row['Risk']}"
        ).add_to(m)

    st_folium(m, width=900, height=420)

    # --- INTERACTIVE TABLE ---
    st.header(L["interactive_table"])
    filter_col1, filter_col2, filter_col3 = st.columns([2,2,1])
    search = filter_col1.text_input("Search city (substring)", "")
    risk_filter = filter_col2.selectbox("Filter by Risk", options=["All","High","Medium","Low"], index=0)
    sort_by = filter_col3.selectbox("Sort by", options=["Victims (desc)","Victims (asc)","City (A-Z)"], index=0)

    table_df = state_df.copy()
    if search:
        table_df = table_df[table_df["City"].str.lower().str.contains(search.lower())]
    if risk_filter != "All":
        table_df = table_df[table_df["Risk"] == risk_filter]
    if sort_by == "Victims (desc)":
        table_df = table_df.sort_values("Victims", ascending=False)
    elif sort_by == "Victims (asc)":
        table_df = table_df.sort_values("Victims", ascending=True)
    else:
        table_df = table_df.sort_values("City", ascending=True)

    if show_table:
        st.dataframe(table_df[["City","Victims","risk_score","Risk","Month","Hour"]].reset_index(drop=True), use_container_width=True)

    # --- CLUSTERING (optional if sklearn available) ---
    if SKLEARN_AVAILABLE:
        try:
            k = min(3, len(state_df))
            if k >= 2:
                km = KMeans(n_clusters=k, random_state=42, n_init="auto")
                state_df["Cluster"] = km.fit_predict(state_df[["Victims"]])
                st.markdown("**Clusters (by Victims)**")
                st.dataframe(state_df[["City","Victims","Cluster"]])
        except Exception:
            pass
    else:
        st.info("Clustering not available (scikit-learn not installed).")

    # --- PROXIMITY CHECK + PANIC MAP HIGHLIGHT ---
    st.header(L["proximity"])
    lat_input = st.number_input("Your latitude", value=float(center_lat))
    lon_input = st.number_input("Your longitude", value=float(center_lon))
    radius_km = st.slider("Alert radius (km)", min_value=1, max_value=50, value=5)

    nearest = None
    for _, row in state_df.iterrows():
        d = haversine(lat_input, lon_input, row["Latitude"], row["Longitude"])
        if nearest is None or d < nearest["dist"]:
            nearest = {"city": row["City"], "dist": d, "risk": row["Risk"], "lat": row["Latitude"], "lon": row["Longitude"]}

    if nearest:
        if nearest["dist"] <= radius_km:
            if nearest["risk"] == "High":
                st.error(f"⚠ You are within {nearest['dist']:.2f} km of HIGH RISK area: {nearest['city']}")
            else:
                st.warning(f"⚠ You are within {nearest['dist']:.2f} km of {nearest['city']} ({nearest['risk']})")
            subm = folium.Map(location=[nearest["lat"], nearest["lon"]], zoom_start=13)
            folium.CircleMarker([lat_input, lon_input], radius=12, color='red', fill=True, fill_color='red').add_to(subm)
            folium.Marker([lat_input, lon_input], popup="You (simulated)").add_to(subm)
            st_folium(subm, width=700, height=300)
        else:
            st.success("✅ You are not near any listed risk zones.")
    else:
        st.info("No nearby zone found.")

# -----------------------
# Route
# -----------------------
if st.session_state.page == "landing":
    landing_page()
else:
    main_page()

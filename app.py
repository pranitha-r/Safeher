# ============================================================
# SafeHer: Smart Women Safety Analytics Website (Streamlit)
# Developed by Nikhila
# ============================================================

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from streamlit_folium import st_folium
import folium
import math
import os
from dotenv import load_dotenv
from twilio.rest import Client

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="SafeHer: Women Safety Analytics", layout="wide")
load_dotenv()

# Optional Twilio setup
TWILIO_ENABLED = os.getenv("TWILIO_ENABLED", "false").lower() == "true"
if TWILIO_ENABLED:
    client = Client(os.getenv("TWILIO_SID"), os.getenv("TWILIO_TOKEN"))
    TWILIO_FROM = os.getenv("TWILIO_FROM")

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def load_data():
    df = pd.read_csv("data/SafeHer_WomenSafety.csv")
    df["Risk"] = df["Victims"].apply(lambda x: "High" if x > 40 else ("Medium" if x > 20 else "Low"))
    return df

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dLat, dLon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ------------------------------
# PAGE HEADER
# ------------------------------
st.title("🛡️ SafeHer: Smart Women Safety Analytics & Alert System")
st.write("An IDS project that uses data analytics to identify unsafe areas and send safety alerts.")

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.title("⚙️ Controls")
user_lat = st.sidebar.number_input("Enter Your Latitude", value=17.3850)
user_lon = st.sidebar.number_input("Enter Your Longitude", value=78.4867)
alert_on = st.sidebar.checkbox("Enable Safety Alert (Distance < 2 km)")

# ------------------------------
# LOAD DATA
# ------------------------------
df = load_data()
st.subheader("📂 Dataset Preview")
st.dataframe(df)

# ------------------------------
# ANALYTICS
# ------------------------------
st.subheader("📊 Crime Analytics by State")
state_summary = df.groupby("State", as_index=False)["Victims"].sum().sort_values("Victims", ascending=False)
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x="Victims", y="State", data=state_summary, palette="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("🥧 Crime Type Distribution")
crime_counts = df["Crime_Type"].value_counts()
fig2, ax2 = plt.subplots()
ax2.pie(crime_counts, labels=crime_counts.index, autopct="%1.1f%%", startangle=90)
ax2.axis("equal")
st.pyplot(fig2)

# ------------------------------
# CLUSTERING
# ------------------------------
st.subheader("🤖 K-Means Clustering")
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
df["Cluster"] = kmeans.fit_predict(df[["Victims"]])
labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
df["Cluster_Label"] = df["Cluster"].map(labels)
st.dataframe(df[["City", "Victims", "Cluster_Label"]])

# ------------------------------
# MAP VISUALIZATION
# ------------------------------
st.subheader("🗺️ Interactive Safety Map")
m = folium.Map(location=[20.59, 78.96], zoom_start=5)
for _, row in df.iterrows():
    color = "red" if row["Risk"] == "High" else ("orange" if row["Risk"] == "Medium" else "green")
    folium.CircleMarker(
        [row["Latitude"], row["Longitude"]],
        radius=8,
        color=color,
        fill=True,
        fill_color=color,
        popup=f"{row['City']} ({row['Risk']}) - {row['Victims']} victims"
    ).add_to(m)
st_data = st_folium(m, width=700, height=500)

# ------------------------------
# SAFETY ALERT SYSTEM
# ------------------------------
st.subheader("📍 Safety Alert System")
nearest = None
for _, row in df.iterrows():
    dist = haversine(user_lat, user_lon, row["Latitude"], row["Longitude"])
    if nearest is None or dist < nearest["dist"]:
        nearest = {"city": row["City"], "risk": row["Risk"], "dist": dist}

if nearest:
    if nearest["dist"] < 2 and alert_on:
        st.error(f"⚠ ALERT! You are near a high-risk area: {nearest['city']} ({nearest['dist']:.2f} km)")
    elif nearest["dist"] < 2:
        st.warning(f"⚠ You are close to {nearest['city']} ({nearest['dist']:.2f} km)")
    else:
        st.success("✅ You are in a safe zone.")

# ------------------------------
# OPTIONAL: SMS ALERT
# ------------------------------
if TWILIO_ENABLED:
    st.subheader("📱 Send Safety SMS")
    phone = st.text_input("Enter phone number (+91...)")
    message = f"SafeHer Alert: You are near {nearest['city']} ({nearest['risk']}) zone."
    if st.button("Send SMS"):
        client.messages.create(to=phone, from_=TWILIO_FROM, body=message)
        st.success("✅ SMS sent successfully!")

# ------------------------------
# INSIGHTS
# ------------------------------
st.subheader("💡 Insights")
st.markdown("""
- **Delhi** shows the highest number of victims — High Risk Zone.  
- **Mumbai** and **Lucknow** are Medium Risk.  
- **Chennai** and **Mysuru** show Low Risk patterns.  
- This data-driven analysis helps identify unsafe areas for women.
""")

st.markdown("---")
st.caption("Developed by pranitha| IDS Project 2025 | Streamlit Website")

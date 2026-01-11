# Page 2 – Geographic Clustering
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

df = pd.read_csv(r"C:\Users\shashank.shandilya_d\Desktop\Chicago Crime\Chicago_Crime_cleaned_data.csv")

st.title("🗺️ Geographic Crime Clusters")

m = folium.Map(location=[41.85, -87.65], zoom_start=10)

for _, row in df.sample(2000).iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=2,
        color=None,
        fill=True,
        fill_color="red",
        fill_opacity=0.4
    ).add_to(m)

st_folium(m, width=1000, height=600)



import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.cluster import KMeans

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(layout="wide")
st.title("🗺️ Geographic Crime Heatmap & Cluster Boundaries")

# -----------------------------------
# LOAD DATA
# -----------------------------------
PROJECT_ROOT = Path(r"C:\Users\shashank.shandilya_d\Desktop\Chicago Crime\Chicago_Crime")
DATA_PATH = PROJECT_ROOT / "data" / "Chicago_Crime_cleaned_data.csv"

if not DATA_PATH.exists():
    st.error(f"❌ Data file not found:\n{DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

# -----------------------------------
# BASIC CLEANING
# -----------------------------------
df = df.dropna(subset=["Latitude", "Longitude"])

# -----------------------------------
# ENSURE CLUSTER COLUMN EXISTS (KEY FIX)
# -----------------------------------
CLUSTER_COL = "Geo_Cluster_KMeans"

if CLUSTER_COL not in df.columns:
    st.warning("⚠️ Cluster column not found. Creating KMeans clusters now...")

    coords = df[["Latitude", "Longitude"]]

    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df[CLUSTER_COL] = kmeans.fit_predict(coords)

    st.success("KMeans clusters created successfully")

# -----------------------------------
# SIDEBAR CONTROLS
# -----------------------------------
st.sidebar.header("🛠 Map Controls")

cluster_options = sorted(df[CLUSTER_COL].unique())

selected_clusters = st.sidebar.multiselect(
    "Select Crime Clusters",
    options=cluster_options,
    default=cluster_options
)

df_filtered = df[df[CLUSTER_COL].isin(selected_clusters)]

# -----------------------------------
# CRIME DENSITY HEATMAP
# -----------------------------------
st.subheader("🔥 Crime Density Heatmap")

heatmap_fig = px.density_mapbox(
    df_filtered,
    lat="Latitude",
    lon="Longitude",
    radius=8,
    center=dict(lat=41.8781, lon=-87.6298),
    zoom=9,
    mapbox_style="carto-positron"
)

st.plotly_chart(heatmap_fig, use_container_width=True)

# -----------------------------------
# CLUSTER OVERLAY MAP
# -----------------------------------
st.subheader("🟢 Geographic Crime Clusters")

cluster_fig = px.scatter_mapbox(
    df_filtered,
    lat="Latitude",
    lon="Longitude",
    color=CLUSTER_COL,
    zoom=9,
    height=650,
    mapbox_style="carto-positron"
)

cluster_fig.update_traces(marker=dict(size=5, opacity=0.6))
st.plotly_chart(cluster_fig, use_container_width=True)

# -----------------------------------
# SUMMARY STATS
# -----------------------------------
st.subheader("📊 Cluster Summary")

summary = (
    df_filtered
    .groupby(CLUSTER_COL)
    .size()
    .reset_index(name="Crime_Count")
)

st.dataframe(summary, use_container_width=True)


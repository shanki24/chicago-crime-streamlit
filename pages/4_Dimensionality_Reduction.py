# Page 4 - PCA + Clustering
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = pd.read_csv("data/Chicago_Crime_cleaned_data.csv")

st.title("📉 PCA + Clustering Visualization")

pca_features = [
    "Hour", "Month", "Day_Num",
    "Latitude", "Longitude",
    "Crime_Severity_Score",
    "District_Crime_Density",
    "Grid_Crime_Density"
]

X = df[pca_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

# KMeans
kmeans = KMeans(n_clusters=6, random_state=42)
df["Geo_Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

# Plot
fig = px.scatter(
    df,
    x="PC1",
    y="PC2",
    color="Geo_Cluster_KMeans",
    title="Crime Clusters in PCA Space",
    opacity=0.6
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### 📊 Explained Variance")
st.write({
    "PC1": round(pca.explained_variance_ratio_[0], 3),
    "PC2": round(pca.explained_variance_ratio_[1], 3),
    "Total": round(pca.explained_variance_ratio_.sum(), 3)
})



import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("📉 Interactive Dimensionality Reduction")

# -------------------------------------------------
# LOAD DATA (SAFE & EXPLICIT)
# -------------------------------------------------
PROJECT_ROOT = Path("data/Chicago_Crime_cleaned_data.csv")
DATA_PATH = PROJECT_ROOT / "data" / "Chicago_Crime_cleaned_data.csv"

if not DATA_PATH.exists():
    st.error(f"❌ Data file not found:\n{DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

# -------------------------------------------------
# FEATURE SELECTION FOR PCA
# -------------------------------------------------
feature_cols = [
    "Lat_Norm", "Lon_Norm",
    "District_Crime_Density",
    "Crime_Severity_Score",
    "Location_Desc_Freq_Norm"
]

df_features = df[feature_cols].dropna()

# -------------------------------------------------
# STANDARDIZE
# -------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.header("🛠 PCA Controls")

n_components = st.sidebar.slider(
    "Number of PCA Components",
    min_value=2,
    max_value=5,
    value=2
)

color_option = st.sidebar.selectbox(
    "Color points by",
    options=["Primary Type", "Geo_Cluster_KMeans"]
)

# -------------------------------------------------
# PCA COMPUTATION
# -------------------------------------------------
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_scaled)

for i in range(n_components):
    df_features[f"PC{i+1}"] = X_pca[:, i]

# Merge back for coloring
df_plot = df.loc[df_features.index].copy()
df_plot = pd.concat([df_plot, df_features[[f"PC{i+1}" for i in range(n_components)]]], axis=1)

# -------------------------------------------------
# PCA SCATTER PLOT
# -------------------------------------------------
st.subheader("🔍 PCA Projection")

fig = px.scatter(
    df_plot,
    x="PC1",
    y="PC2",
    color=color_option,
    opacity=0.6,
    title="Crime Data in PCA Space",
    hover_data=["Primary Type", "District", "Year"]
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# EXPLAINED VARIANCE
# -------------------------------------------------
st.subheader("📊 Explained Variance")

var_df = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(n_components)],
    "Explained Variance (%)": pca.explained_variance_ratio_ * 100
})

fig_var = px.bar(
    var_df,
    x="Component",
    y="Explained Variance (%)",
    text_auto=".2f",
    title="Explained Variance by Principal Components"
)

st.plotly_chart(fig_var, use_container_width=True)

# -------------------------------------------------
# INTERPRETATION
# -------------------------------------------------
st.subheader("🧠 Interpretation")

st.markdown("""
- **Each point** represents a crime incident
- **Distance between points** reflects similarity
- **Clusters indicate crime behavior patterns**
- PCA helps reduce noise while preserving structure
""")

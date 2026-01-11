# Page 5 – Model Comparison
import streamlit as st
import pandas as pd

st.title("🏆 Model Comparison")

results = pd.DataFrame({
    "Model": ["KMeans", "DBSCAN", "PCA + KMeans"],
    "Silhouette Score": [0.39, 0.41, 0.58]
})

st.dataframe(results, use_container_width=True)

best_model = results.sort_values("Silhouette Score", ascending=False).iloc[0]
st.success(f"Best Model: {best_model['Model']} (Score: {best_model['Silhouette Score']})")



import streamlit as st
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("📊 Model Performance Monitoring (MLflow)")

# -------------------------------------------------
# MLflow CONFIG (LOCAL)
# -------------------------------------------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

st.success(f"Connected to MLflow at {MLFLOW_TRACKING_URI}")

# -------------------------------------------------
# FETCH EXPERIMENTS
# -------------------------------------------------
experiments = client.search_experiments()

if not experiments:
    st.error("❌ No MLflow experiments found.")
    st.stop()

exp_names = {exp.name: exp.experiment_id for exp in experiments}

# -------------------------------------------------
# SIDEBAR – EXPERIMENT SELECTION
# -------------------------------------------------
st.sidebar.header("🧪 Experiment Selection")

selected_exp_name = st.sidebar.selectbox(
    "Select MLflow Experiment",
    options=list(exp_names.keys())
)

experiment_id = exp_names[selected_exp_name]

# -------------------------------------------------
# FETCH RUNS
# -------------------------------------------------
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.silhouette_score DESC"]
)

if not runs:
    st.warning("No runs found in this experiment.")
    st.stop()

# -------------------------------------------------
# BUILD METRICS TABLE
# -------------------------------------------------
rows = []

for run in runs:
    row = {
        "Run ID": run.info.run_id,
        "Run Name": run.data.tags.get("mlflow.runName", "N/A"),
        "Algorithm": run.data.params.get("algorithm", "N/A"),
        "Clusters": run.data.params.get("n_clusters", "N/A"),
        "Silhouette Score": run.data.metrics.get("silhouette_score", None),
        "Explained Variance": run.data.metrics.get("cumulative_variance", None)
    }
    rows.append(row)

df_runs = pd.DataFrame(rows)

# -------------------------------------------------
# SUMMARY
# -------------------------------------------------
st.subheader("📌 Experiment Summary")

st.dataframe(df_runs, use_container_width=True)

# -------------------------------------------------
# BEST MODEL
# -------------------------------------------------
best_run = df_runs.sort_values(
    by="Silhouette Score",
    ascending=False
).iloc[0]

st.subheader("🏆 Best Performing Model")

col1, col2, col3 = st.columns(3)

col1.metric("Algorithm", best_run["Algorithm"])
col2.metric("Silhouette Score", round(best_run["Silhouette Score"], 4))
col3.metric("Clusters", best_run["Clusters"])

# -------------------------------------------------
# PERFORMANCE COMPARISON
# -------------------------------------------------
st.subheader("📈 Model Performance Comparison")

chart_df = df_runs.dropna(subset=["Silhouette Score"])

st.bar_chart(
    chart_df.set_index("Run Name")["Silhouette Score"]
)

# -------------------------------------------------
# DETAILED RUN INSPECTION
# -------------------------------------------------
st.subheader("🔍 Inspect Individual Run")

selected_run_id = st.selectbox(
    "Select Run ID",
    options=df_runs["Run ID"].tolist()
)

selected_run = client.get_run(selected_run_id)

st.markdown("### Parameters")
st.json(selected_run.data.params)

st.markdown("### Metrics")
st.json(selected_run.data.metrics)

st.markdown("### Tags")
st.json(selected_run.data.tags)

# -------------------------------------------------
# INTERPRETATION
# -------------------------------------------------
st.subheader("🧠 Monitoring Insights")

st.markdown("""
- **Higher silhouette score → better cluster separation**
- Compare PCA vs raw KMeans runs
- Track performance drift across experiments
- Identify best production-ready model
""")

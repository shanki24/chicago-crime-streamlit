# Page 1 - Crime Overview
import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("data/Chicago_Crime_cleaned_data.csv")

st.title("📊 Crime Overview")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        df,
        x="Primary Type",
        title="Crime Type Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        df,
        x="Hour",
        title="Crimes by Hour"
    )
    st.plotly_chart(fig, use_container_width=True)

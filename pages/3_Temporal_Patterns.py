# Page 3 – Temporal Patterns
import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("data/Chicago_Crime_cleaned_data.csv")

st.title("⏰ Temporal Crime Patterns")

fig = px.line(
    df.groupby("Hour").size().reset_index(name="Crimes"),
    x="Hour",
    y="Crimes",
    title="Crimes by Hour"
)

st.plotly_chart(fig, use_container_width=True)


import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(layout="wide")
st.title("⏰ Temporal Crime Pattern Analysis")

# -------------------------------
# LOAD DATA (SAFE & PORTABLE)
# -------------------------------
DATA_PATH = Path("data") / "Chicago_Crime_cleaned_data.csv"

if not DATA_PATH.exists():
    st.error(f"❌ Data file not found:\n{DATA_PATH.resolve()}")
    st.stop()

df = pd.read_csv(DATA_PATH)

# -----------------------------------
# DATE HANDLING
# -----------------------------------
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

# -----------------------------------
# SIDEBAR FILTERS
# -----------------------------------
st.sidebar.header("🛠 Filters")

year_min = int(df["Year"].min())
year_max = int(df["Year"].max())

if year_min == year_max:
    # Only one year available → no range slider
    st.sidebar.info(f"Only data available for year {year_min}")
    year_range = (year_min, year_max)
else:
    year_range = st.sidebar.slider(
        "Select Year Range",
        year_min,
        year_max,
        (year_min, year_max)
    )

df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

# -----------------------------------
# ROW 1 — HOURLY + WEEKDAY
# -----------------------------------
st.subheader("🕐 Hourly & Weekly Crime Trends")

col1, col2 = st.columns(2)

with col1:
    hourly_counts = df.groupby("Hour").size().reset_index(name="Crimes")
    fig_hour = px.line(
        hourly_counts,
        x="Hour",
        y="Crimes",
        markers=True,
        title="Crimes by Hour of Day"
    )
    st.plotly_chart(fig_hour, use_container_width=True)

with col2:
    weekday_counts = df.groupby("Day_of_Week").size().reset_index(name="Crimes")
    weekday_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]
    fig_weekday = px.bar(
        weekday_counts,
        x="Day_of_Week",
        y="Crimes",
        category_orders={"Day_of_Week": weekday_order},
        title="Crimes by Day of Week"
    )
    st.plotly_chart(fig_weekday, use_container_width=True)

# -----------------------------------
# ROW 2 — MONTHLY & WEEKEND
# -----------------------------------
st.subheader("📅 Monthly & Weekend Patterns")

col3, col4 = st.columns(2)

with col3:
    month_counts = df.groupby("Month").size().reset_index(name="Crimes")
    fig_month = px.line(
        month_counts,
        x="Month",
        y="Crimes",
        markers=True,
        title="Crimes by Month"
    )
    st.plotly_chart(fig_month, use_container_width=True)

with col4:
    weekend_counts = df.groupby("Is_Weekend").size().reset_index(name="Crimes")
    weekend_counts["Is_Weekend"] = weekend_counts["Is_Weekend"].map(
        {0: "Weekday", 1: "Weekend"}
    )
    fig_weekend = px.pie(
        weekend_counts,
        names="Is_Weekend",
        values="Crimes",
        title="Weekend vs Weekday Crimes"
    )
    st.plotly_chart(fig_weekend, use_container_width=True)

# -----------------------------------
# ROW 3 — YEARLY TREND
# -----------------------------------
st.subheader("📈 Long-Term Crime Trend")

yearly_counts = df.groupby("Year").size().reset_index(name="Crimes")

fig_year = px.line(
    yearly_counts,
    x="Year",
    y="Crimes",
    markers=True,
    title="Yearly Crime Trend"
)

st.plotly_chart(fig_year, use_container_width=True)

# -----------------------------------
# SUMMARY INSIGHTS
# -----------------------------------
st.subheader("🧠 Key Insights")

peak_hour = hourly_counts.loc[hourly_counts["Crimes"].idxmax(), "Hour"]
busiest_day = weekday_counts.loc[weekday_counts["Crimes"].idxmax(), "Day_of_Week"]
busiest_month = month_counts.loc[month_counts["Crimes"].idxmax(), "Month"]

st.markdown(f"""
- 🔥 **Peak crime hour:** {int(peak_hour)}:00
- 📆 **Busiest day:** {busiest_day}
- 🌡️ **Highest crime month:** {int(busiest_month)}
""")

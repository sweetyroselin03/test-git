import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from datetime import datetime, timedelta

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="TNEB Smart Grid Analytics",
    layout="wide",
    page_icon="⚡"
)

st.title("⚡ Tamil Nadu Electricity Board – Smart Grid Analytics Platform")
st.markdown(
    "**Real-Time Electricity Demand Monitoring & Grid Intelligence**  \n"
    "*Tamil Nadu Generation and Distribution Corporation (TANGEDCO)*"
)

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("⚙️ TNEB Control Panel")

auto_refresh = st.sidebar.checkbox("Enable Live SCADA Refresh (5s)", True)

alert_threshold = st.sidebar.slider(
    "Critical Load Threshold (MW)",
    120, 200, 150
)

region_filter = st.sidebar.multiselect(
    "Select Region",
    ["North TN", "South TN", "West TN", "Central TN", "Delta TN", "Coastal TN"],
    default=["North TN", "South TN", "West TN"]
)

# --------------------------------------------------
# SINGLE MASTER DATASET (TN District → Region Mapping)
# --------------------------------------------------
@st.cache_data(ttl=10)
def create_master_dataset(threshold):
    # District → Region mapping
    district_region = {
        "Chennai": "North TN",
        "Tiruvallur": "North TN",
        "Kancheepuram": "North TN",
        "Vellore": "North TN",
        "Ranipet": "North TN",

        "Madurai": "South TN",
        "Tirunelveli": "South TN",
        "Thoothukudi": "South TN",
        "Kanyakumari": "South TN",

        "Coimbatore": "West TN",
        "Erode": "West TN",
        "Salem": "West TN",
        "Tiruppur": "West TN",

        "Trichy": "Central TN",
        "Karur": "Central TN",
        "Namakkal": "Central TN",
        "Perambalur": "Central TN",

        "Thanjavur": "Delta TN",
        "Tiruvarur": "Delta TN",
        "Nagapattinam": "Delta TN",
        "Mayiladuthurai": "Delta TN",

        "Cuddalore": "Coastal TN",
        "Villupuram": "Coastal TN",
        "Chengalpattu": "Coastal TN"
    }

    n = 1200
    timestamps = pd.date_range("2026-01-26", periods=n, freq="5min")

    districts = np.random.choice(list(district_region.keys()), n)
    regions = [district_region[d] for d in districts]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "district": districts,
        "region": regions,
        "load_mw": (
            90
            + 40 * np.sin(np.linspace(0, 5 * np.pi, n))
            + np.random.normal(0, 10, n)
        )
    })

    # Feature Engineering
    df["hour"] = df["timestamp"].dt.hour
    df["rolling_1h"] = df["load_mw"].rolling(12).mean()
    df["peak_flag"] = df["load_mw"] >= threshold

    z = (df["load_mw"] - df["load_mw"].mean()) / df["load_mw"].std()
    df["z_score"] = z
    df["anomaly"] = z.abs() > 2

    return df

# Load dataset
df = create_master_dataset(alert_threshold)
df = df[df["region"].isin(region_filter)]

# --------------------------------------------------
# Module 1: Regional Load Trends
# --------------------------------------------------
st.header("📈 Module 1: Regional Load Trend Analysis")

col1, col2 = st.columns(2)

with col1:
    trend_fig = px.line(
        df,
        x="timestamp",
        y="rolling_1h",
        color="region",
        title="Rolling 1-Hour Average Load by Region (MW)"
    )
    st.plotly_chart(trend_fig, use_container_width=True)

with col2:
    heatmap_data = (
        df.groupby(["hour", "region"])["load_mw"]
        .mean()
        .reset_index()
    )

    heatmap_fig = px.imshow(
        heatmap_data.pivot(index="region", columns="hour", values="load_mw"),
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Average Hourly Load Pattern – Tamil Nadu Regions"
    )
    st.plotly_chart(heatmap_fig, use_container_width=True)

# --------------------------------------------------
# Module 2: Peak Load Analysis
# --------------------------------------------------
st.header("🔴 Module 2: Peak Load Identification & Grid Stress")

peaks = df[df["load_mw"] >= alert_threshold]

st.dataframe(
    peaks[["timestamp", "district", "region", "load_mw"]]
    .sort_values("load_mw", ascending=False)
    .head(15),
    use_container_width=True
)

peak_fig = px.bar(
    peaks.tail(20),
    x="timestamp",
    y="load_mw",
    color="region",
    title="Recent Peak Load Events Across Tamil Nadu"
)
st.plotly_chart(peak_fig, use_container_width=True)

# --------------------------------------------------
# Module 3: District & Region Benchmarking
# --------------------------------------------------
st.header("🚨 Module 3: District & Region Performance Benchmarking")

kpis = (
    df.groupby(["region", "district"])
    .agg(
        Avg_Load_MW=("load_mw", "mean"),
        Max_Load_MW=("load_mw", "max"),
        Volatility_SD=("load_mw", "std"),
        Peak_Events=("peak_flag", "sum")
    )
    .round(2)
)

st.dataframe(kpis, use_container_width=True)

anomaly_fig = px.scatter(
    df[df["anomaly"]].tail(200),
    x="timestamp",
    y="load_mw",
    color="region",
    hover_data=["district"],
    size=df[df["anomaly"]]["z_score"].abs(),
    title="Detected Load Anomalies Across Tamil Nadu Districts"
)
st.plotly_chart(anomaly_fig, use_container_width=True)

# --------------------------------------------------
# Module 4: Live Load Dispatch Monitoring
# --------------------------------------------------
st.header("🖥️ Module 4: Live Load Dispatch Centre Monitoring (TNEB)")

placeholder = st.empty()
alerts = st.empty()

if auto_refresh:
    for _ in range(20):
        with placeholder.container():
            live_df = df.tail(50).copy()
            live_df["timestamp"] = pd.date_range(
                start=datetime.now() - timedelta(minutes=250),
                periods=50,
                freq="5min"
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⚡ Current Load (MW)", f"{live_df['load_mw'].iloc[-1]:.1f}")
            with col2:
                st.metric("📈 Peak Load (MW)", f"{live_df['load_mw'].max():.1f}")
            with col3:
                st.metric("🚨 Active Anomalies", live_df["anomaly"].sum())

            live_fig = px.line(
                live_df,
                x="timestamp",
                y="load_mw",
                color="region",
                title="Live Regional Load Monitoring"
            )
            st.plotly_chart(live_fig, use_container_width=True)

        recent_peaks = live_df[live_df["load_mw"] >= alert_threshold]
        if not recent_peaks.empty:
            with alerts.container():
                st.error(
                    f"🚨 **TNEB GRID ALERT** | "
                    f"{recent_peaks.iloc[-1]['district']} – "
                    f"{recent_peaks.iloc[-1]['load_mw']:.1f} MW at "
                    f"{recent_peaks.iloc[-1]['timestamp'].strftime('%H:%M:%S')}"
                )

        time.sleep(5)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "**Tamil Nadu Electricity Board (TNEB) / TANGEDCO – Analytics Prototype**  \n"
    "*State-Level Smart Grid Monitoring & Decision Support System*"
)

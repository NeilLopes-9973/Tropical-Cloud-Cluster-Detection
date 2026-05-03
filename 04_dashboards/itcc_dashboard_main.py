"""
Original ITCC Dashboard with Animation Features
INSAT-3D Tropical Cloud Cluster Tracking - Dark Theme with Time-lapse Animation

Author: Neil Lopes | Version: 1.0 | Date: October 2025
Features: Dark Theme, Animation Play/Pause, Real-time Filtering, Track Analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION & DARK THEME
# ============================================================================

st.set_page_config(
    page_title="ITCC Dashboard - Animation",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS - Target Streamlit Root Elements
st.markdown("""
<style>
/* Target Streamlit root elements */
html, body {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #16213e 100%);
    color: #90caf9;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #16213e 100%);
    padding: 0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(100, 181, 246, 0.2);
}

header {visibility: hidden;}
.stApp {margin-top: -50px;}
[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    padding: 1rem 2rem 2rem 2rem;
    max-width: 100% !important;
    width: 100% !important;
}

/* Headers */
.main-header {
    font-size: 2.8rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 1rem;
    background: linear-gradient(90deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.subtitle {
    font-size: 1.3rem;
    color: #64b5f6;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* Metrics */
.stMetric {
    background-color: rgba(13, 27, 42, 0.9) !important;
    border: 1px solid rgba(100, 181, 246, 0.3);
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
}

.stMetric:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(100, 181, 246, 0.2);
}

.stMetric label {
    color: #64b5f6 !important;
    font-weight: 600;
    font-size: 0.9rem;
}

/* Text elements */
h1, h2, h3, h4, h5, h6 {
    color: #90caf9 !important;
    font-weight: 600;
}

p, span, div {
    color: #b0bec5 !important;
}

/* DataFrames */
.stDataFrame {
    background-color: rgba(13, 27, 42, 0.7) !important;
    border: 1px solid rgba(100, 181, 246, 0.2);
    border-radius: 8px;
}

/* Plotly dark theme */
.js-plotly-plot .plotly {
    background: transparent !important;
}

.plotly .plot-container {
    background: transparent !important;
}

/* Form elements */
.stSelectbox > div > div > select,
.stSlider > div > div > div,
.stDateInput > div > div > input,
.stTextInput > div > div > input {
    background-color: rgba(13, 27, 42, 0.8) !important;
    color: #90caf9 !important;
    border: 1px solid rgba(100, 181, 246, 0.3);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #1565c0 0%, #0d47a1 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1976d2 0%, #1565c0 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(25, 118, 210, 0.3);
}

/* Expander */
.streamlit-expanderHeader {
    background-color: rgba(13, 27, 42, 0.8) !important;
    border: 1px solid rgba(100, 181, 246, 0.3);
}

.streamlit-expanderContent {
    background-color: rgba(13, 27, 42, 0.6) !important;
}

/* Animation controls */
.animation-controls {
    background: rgba(13, 27, 42, 0.9);
    border: 1px solid rgba(100, 181, 246, 0.3);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

/* Footer */
.dark-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: rgba(13, 27, 42, 0.95);
    color: #64b5f6;
    text-align: center;
    padding: 12px;
    border-top: 2px solid rgba(100, 181, 246, 0.3);
    font-size: 13px;
    backdrop-filter: blur(10px);
}

/* Status indicators */
.status-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 2s infinite;
}

.status-playing {
    background-color: #4caf50;
}

.status-paused {
    background-color: #ff9800;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

/* Remove white backgrounds */
element.style {
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# Force full width for main section
st.markdown("""
<style>
section.main > div {
    max-width: 100% !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=1800)
def load_itcc_data():
    """Load ITCC dataset with animation support"""
    data_paths = [
        Path("../data/final/ITCC_v2_cleaned.csv"),
        Path("../data/final/ITCC_v2.csv"),
        Path("data/final/ITCC_v2_cleaned.csv"),
        Path("data/final/ITCC_v2.csv")
    ]
    
    for path in data_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                # Convert timestamp
                if 'Time_UTC' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['Time_UTC'])
                elif 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except Exception as e:
                st.warning(f"Error loading {path}: {e}")
                continue
    
    # Generate sample data with animation-friendly structure
    st.warning("No ITCC dataset found. Using sample data with animation support.")
    return generate_animated_sample_data()

def generate_animated_sample_data():
    """Generate sample data optimized for animation"""
    np.random.seed(42)
    tracks = []
    base_date = datetime(2024, 9, 20)
    
    for track_id in range(1, 31):  # 30 tracks for smoother animation
        n_obs = np.random.randint(20, 35)
        
        for i in range(n_obs):
            timestamp = base_date + timedelta(hours=i*0.5)
            
            # Indian Ocean region with smooth movement
            lat = 12.5 + np.sin(i*0.1) * 3 + track_id * 0.2
            lon = 75.0 + np.cos(i*0.1) * 5 + track_id * 0.3
            
            # Smooth property evolution
            area_km2 = 1000 + np.sin(i*0.15) * 500 + track_id * 50
            tb_min = 210 + np.sin(i*0.2) * 15 + track_id * 0.5
            tb_mean = tb_min + np.random.uniform(10, 20)
            tb_max = tb_mean + np.random.uniform(5, 15)
            cth_max_km = 12 + np.sin(i*0.1) * 2
            
            # Lifecycle progression
            if i < n_obs * 0.3:
                stage = 'Developing'
            elif i < n_obs * 0.7:
                stage = 'Mature'
            else:
                stage = 'Dissipating'
            
            # Quality flag
            quality = np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05])
            
            tracks.append({
                'Track_ID': track_id,
                'Time_UTC': timestamp,
                'timestamp': timestamp,
                'Lat': lat,
                'Lon': lon,
                'Area_km2': area_km2,
                'Tb_min': tb_min,
                'Tb_mean': tb_mean,
                'Tb_max': tb_max,
                'CTH_max_km': cth_max_km,
                'Lifecycle_Stage': stage,
                'quality_flag': quality,
                'frame_index': i,
                'sequence_id': track_id
            })

    return pd.DataFrame(tracks)

def create_animated_map(df):
    """Create presentation-ready animated cloud visualization over the Indian Ocean."""
    indian_ocean_bounds = dict(lat_min=-30, lat_max=30, lon_min=30, lon_max=110)

    def empty_map(message):
        fig = go.Figure()
        fig.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=5, lon=75),
                zoom=3.7,
                bounds=dict(
                    west=indian_ocean_bounds["lon_min"],
                    east=indian_ocean_bounds["lon_max"],
                    south=indian_ocean_bounds["lat_min"],
                    north=indian_ocean_bounds["lat_max"]
                )
            ),
            height=650,
            autosize=True,
            margin=dict(l=0, r=0, t=28, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E3F2FD", size=13),
            annotations=[dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="#E3F2FD", size=18)
            )]
        )
        return fig

    if df is None or len(df) == 0:
        return empty_map("No cloud clusters available")

    required_cols = ["Lat", "Lon"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return empty_map("Latitude/Longitude columns missing")

    df = df.copy()
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Lon"] = pd.to_numeric(df["Lon"], errors="coerce")
    df = df.dropna(subset=["Lat", "Lon"])

    df = df[
        (df["Lat"].between(indian_ocean_bounds["lat_min"], indian_ocean_bounds["lat_max"])) &
        (df["Lon"].between(indian_ocean_bounds["lon_min"], indian_ocean_bounds["lon_max"]))
    ].copy()

    if df.empty:
        return empty_map("No valid Indian Ocean points")

    time_col = "timestamp" if "timestamp" in df.columns else None
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col)
        df["time_str"] = df[time_col].dt.strftime("%Y-%m-%d %H:%M")
    else:
        df["time_str"] = "Current"

    if df.empty:
        return empty_map("No valid animation frames")

    if "Area_km2" not in df.columns:
        df["Area_km2"] = 4000
    df["Area_km2"] = pd.to_numeric(df["Area_km2"], errors="coerce").fillna(4000).clip(lower=500)

    if "Tb_min" not in df.columns:
        df["Tb_min"] = 220
    df["Tb_min"] = pd.to_numeric(df["Tb_min"], errors="coerce").fillna(220)

    hover_cols = [col for col in ["Track_ID", "Lifecycle_Stage", "Area_km2", "Tb_min", "Lat", "Lon"] if col in df.columns]

    fig = px.scatter_mapbox(
        df,
        lat="Lat",
        lon="Lon",
        color="Tb_min",
        size="Area_km2",
        animation_frame="time_str",
        color_continuous_scale=["#E3F2FD", "#4FC3F7", "#0288D1", "#01579B"],
        size_max=24,
        zoom=3.7,
        center=dict(lat=5, lon=75),
        height=650,
        hover_data=hover_cols
    )

    fig.update_traces(
        marker=dict(opacity=0.82, sizemin=5),
        selector=dict(type="scattermapbox")
    )

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=5, lon=75),
            zoom=3.7,
            bounds=dict(
                west=indian_ocean_bounds["lon_min"],
                east=indian_ocean_bounds["lon_max"],
                south=indian_ocean_bounds["lat_min"],
                north=indian_ocean_bounds["lat_max"]
            )
        ),
        title=dict(
            text="Indian Ocean Cloud Cluster Animation",
            x=0.02,
            font=dict(color="#E3F2FD", size=20)
        ),
        height=650,
        autosize=True,
        margin=dict(l=0, r=0, t=44, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E3F2FD", size=13),
        legend=dict(
            title="",
            bgcolor="rgba(5, 12, 24, 0.72)",
            bordercolor="rgba(227, 242, 253, 0.22)",
            borderwidth=1,
            font=dict(color="#E3F2FD")
        ),
        coloraxis_colorbar=dict(
            title=dict(text="Min Tb (K)", font=dict(color="#E3F2FD")),
            thickness=14,
            len=0.72,
            bgcolor="rgba(5, 12, 24, 0.72)",
            tickfont=dict(color="#E3F2FD")
        )
    )

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="left",
            active=-1,
            x=0.02,
            y=0.02,
            xanchor="left",
            yanchor="bottom",
            bgcolor="#1565c0",
            bordercolor="#00e5ff",
            font=dict(color="#ffffff"),
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=300, redraw=True), transition=dict(duration=100), fromcurrent=True)]
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]
                )
            ]
        )],
        hoverlabel=dict(
            bgcolor="#0a0e27",
            font_size=12,
            font_family="Arial"
        )
    )

    return fig

def create_animated_trajectory_map(df_track):
    """Create animated cumulative trajectory for the selected filtered track."""
    indian_ocean_bounds = dict(lat_min=-30, lat_max=30, lon_min=30, lon_max=110)

    def empty_map(message):
        fig = go.Figure()
        fig.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=5, lon=75),
                zoom=3.8,
                bounds=dict(
                    west=indian_ocean_bounds["lon_min"],
                    east=indian_ocean_bounds["lon_max"],
                    south=indian_ocean_bounds["lat_min"],
                    north=indian_ocean_bounds["lat_max"]
                )
            ),
            height=650,
            autosize=True,
            margin=dict(l=0, r=0, t=44, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E3F2FD", size=13),
            annotations=[dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="#E3F2FD", size=18)
            )]
        )
        return fig

    if df_track is None or len(df_track) == 0:
        return empty_map("No trajectory data available")

    required_cols = ["Lat", "Lon"]
    missing_cols = [col for col in required_cols if col not in df_track.columns]
    if missing_cols:
        return empty_map("Latitude/Longitude columns missing")

    df_track = df_track.copy()
    df_track["Lat"] = pd.to_numeric(df_track["Lat"], errors="coerce")
    df_track["Lon"] = pd.to_numeric(df_track["Lon"], errors="coerce")
    df_track = df_track.dropna(subset=["Lat", "Lon"])

    df_track = df_track[
        (df_track["Lat"].between(indian_ocean_bounds["lat_min"], indian_ocean_bounds["lat_max"])) &
        (df_track["Lon"].between(indian_ocean_bounds["lon_min"], indian_ocean_bounds["lon_max"]))
    ].copy()

    if df_track.empty:
        return empty_map("No valid Indian Ocean trajectory points")

    if "timestamp" in df_track.columns:
        df_track["timestamp"] = pd.to_datetime(df_track["timestamp"], errors="coerce")
        df_track = df_track.dropna(subset=["timestamp"]).sort_values("timestamp")
        df_track["time_label"] = df_track["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    elif "Time_UTC" in df_track.columns:
        df_track["Time_UTC"] = pd.to_datetime(df_track["Time_UTC"], errors="coerce")
        df_track = df_track.dropna(subset=["Time_UTC"]).sort_values("Time_UTC")
        df_track["time_label"] = df_track["Time_UTC"].dt.strftime("%Y-%m-%d %H:%M")
    else:
        df_track = df_track.reset_index(drop=True)
        df_track["time_label"] = "Step " + (df_track.index + 1).astype(str)

    df_track = df_track.reset_index(drop=True)
    df_track["step"] = np.arange(len(df_track))

    if df_track.empty:
        return empty_map("No valid trajectory frames")

    if len(df_track) <= 1:
        return empty_map("No trajectory data available")

    lat_min = df_track["Lat"].min()
    lat_max = df_track["Lat"].max()
    lon_min = df_track["Lon"].min()
    lon_max = df_track["Lon"].max()

    center_lat = df_track["Lat"].mean()
    center_lon = df_track["Lon"].mean()

    max_range = max(lat_max - lat_min, lon_max - lon_min)
    if max_range < 2:
        zoom = 6
    elif max_range < 5:
        zoom = 5
    else:
        zoom = 4

    def cumulative_trace(frame_index):
        frame_data = df_track.iloc[:frame_index]
        return go.Scattermapbox(
            lat=frame_data["Lat"],
            lon=frame_data["Lon"],
            mode="lines+markers",
            line=dict(color="cyan", width=4),
            marker=dict(size=7, color="white"),
            name="Trajectory Path",
            customdata=np.stack([frame_data["time_label"]], axis=-1),
            hovertemplate="Time: %{customdata[0]}<br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>"
        )

    start_point = df_track.iloc[0]
    end_point = df_track.iloc[-1]

    fig = go.Figure(
        data=[
            go.Scattermapbox(
                lat=[],
                lon=[],
                mode="lines+markers",
                line=dict(color="cyan", width=4),
                marker=dict(size=7, color="white"),
                name="Trajectory Path",
                hovertemplate="Time: %{customdata[0]}<br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>"
            )
        ],
        frames=[
            go.Frame(
                data=[cumulative_trace(i)],
                traces=[0],
                name=str(i)
            )
            for i in range(1, len(df_track) + 1)
        ]
    )

    fig.add_trace(go.Scattermapbox(
        lat=[start_point["Lat"]],
        lon=[start_point["Lon"]],
        mode="markers",
        marker=dict(size=16, color="#00E676"),
        name="Start",
        hovertemplate="Start<br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scattermapbox(
        lat=[end_point["Lat"]],
        lon=[end_point["Lon"]],
        mode="markers",
        marker=dict(size=16, color="#FF1744"),
        name="End",
        hovertemplate="End<br>Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<extra></extra>"
    ))

    slider_steps = [
        dict(
            method="animate",
            label=str(i),
            args=[[str(i)], dict(mode="immediate", frame=dict(duration=300, redraw=True), transition=dict(duration=100))]
        )
        for i in range(1, len(df_track) + 1)
    ]

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        title=dict(
            text="Track Trajectory - Full Lifecycle",
            x=0.02,
            font=dict(color="#E3F2FD", size=20)
        ),
        height=650,
        autosize=True,
        margin=dict(l=0, r=0, t=44, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E3F2FD", size=13),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(5, 12, 24, 0.72)",
            bordercolor="rgba(227, 242, 253, 0.22)",
            borderwidth=1,
            font=dict(color="#E3F2FD")
        ),
        updatemenus=[{
            "type": "buttons",
            "direction": "left",
            "active": -1,
            "x": 0.02,
            "y": 0.02,
            "xanchor": "left",
            "yanchor": "bottom",
            "bgcolor": "#1565c0",
            "bordercolor": "#00e5ff",
            "font": {"color": "#ffffff"},
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 300, "redraw": True}, "transition": {"duration": 100}, "fromcurrent": True}]
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                }
            ]
        }],
        sliders=[dict(
            active=0,
            x=0.15,
            y=0.035,
            len=0.80,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(5, 12, 24, 0.72)",
            bordercolor="rgba(227, 242, 253, 0.22)",
            borderwidth=1,
            currentvalue=dict(prefix="Step: ", font=dict(color="#E3F2FD", size=13)),
            steps=slider_steps
        )],
        hoverlabel=dict(
            bgcolor="#0a0e27",
            font_size=12,
            font_family="Arial"
        )
    )

    return fig

def create_dual_axis_track_evolution(df_track):
    df_track = df_track.copy()
    df_track["timestamp"] = pd.to_datetime(df_track["timestamp"], errors="coerce")
    df_track["Area_km2"] = pd.to_numeric(df_track["Area_km2"], errors="coerce")
    df_track["Tb_min"] = pd.to_numeric(df_track["Tb_min"], errors="coerce")
    df_track = df_track.dropna(subset=["timestamp", "Area_km2", "Tb_min"]).sort_values("timestamp")

    # Remove duplicate timestamps (fix vertical lines) without aggregating non-numeric columns
    df_track = df_track.groupby("timestamp", as_index=False).agg({"Area_km2": "mean", "Tb_min": "mean"})

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df_track["timestamp"],
            y=df_track["Area_km2"],
            mode="lines+markers",
            line=dict(color="#00BCD4", width=3),
            marker=dict(size=6),
            name="Area (km²)",
            hovertemplate="Time: %{x|%d %B %Y %H:%M}<br>Area: %{y:.0f} km²<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_track["timestamp"],
            y=df_track["Tb_min"],
            mode="lines+markers",
            line=dict(color="#F44336", width=2),
            marker=dict(size=5),
            name="Tb_min (K)",
            hovertemplate="Time: %{x|%d %B %Y %H:%M}<br>Tb_min: %{y:.1f} K<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        height=500,
        plot_bgcolor="#0a0e27",
        paper_bgcolor="#0a0e27",
        font=dict(color="#E0E0E0"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.05),
        margin=dict(l=40, r=40, t=40, b=40),
    )

    fig.update_xaxes(title="Date", tickformat="%d %b %Y", showgrid=False)
    fig.update_yaxes(
        title_text="Area (km²)",
        color="#00BCD4",
        secondary_y=False,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        dtick=5000,
    )
    fig.update_yaxes(title_text="Tb_min (K)", color="#F44336", secondary_y=True)
    fig.update_yaxes(secondary_y=True, showgrid=False)

    return fig

def create_time_series(df_track):
    """Create time series with animation indicators"""
    df_track_sorted = df_track.sort_values('timestamp')
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('📏 Cluster Area Evolution', '🌡️ Temperature Evolution'),
        vertical_spacing=0.12
    )
    
    # Area evolution with gradient
    fig.add_trace(
        go.Scatter(
            x=df_track_sorted['timestamp'],
            y=df_track_sorted['Area_km2'],
            mode='lines+markers',
            line=dict(color='#64b5f6', width=3),
            marker=dict(
                size=8,
                color=df_track_sorted['frame_index'],
                colorscale='Blues',
                showscale=False,
                colorbar=dict(title="Frame", x=1.02)
            ),
            name='Area (km2)',
            hovertemplate='Date: %{x|%d %B %Y}<br>Time: %{x|%H:%M} UTC<br>Area: %{y:.0f} km2<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Temperature evolution
    fig.add_trace(
        go.Scatter(
            x=df_track_sorted['timestamp'],
            y=df_track_sorted['Tb_min'],
            mode='lines+markers',
            line=dict(color='#ff7043', width=3),
            marker=dict(
                size=8,
                color=df_track_sorted['frame_index'],
                colorscale='Reds',
                showscale=False
            ),
            name='Tb_min (K)',
            hovertemplate='Date: %{x|%d %B %Y}<br>Time: %{x|%H:%M} UTC<br>Tb_min: %{y:.1f} K<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor='rgba(13, 27, 42, 0.8)',
        paper_bgcolor='rgba(13, 27, 42, 0.8)',
        font=dict(color='#90caf9')
    )
    
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)
    fig.update_yaxes(title_text="Area (km2)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (K)", row=2, col=1)
    
    return fig

# ============================================================================
# MAIN APPLICATION WITH ANIMATION
# ============================================================================

# Load data
with st.spinner("🌊 Loading ITCC dataset with animation support..."):
    df = load_itcc_data()

# Header
st.markdown('<div class="main-header">🌊 ITCC Animated Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">INSAT-3D Tropical Cloud Cluster Tracking - Time-lapse Animation</div>', unsafe_allow_html=True)

# Dataset Information Section
with st.expander("📊 Dataset Information", expanded=False):
    if 'timestamp' in df.columns:
        total_records = len(df)
        date_range = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
        unique_tracks = df['Track_ID'].nunique() if 'Track_ID' in df.columns else 'N/A'
        available_dates = sorted(df['timestamp'].dt.date.unique())
        observation_dates = f"{len(available_dates)} dates available"
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Total Records", f"{total_records:,}")
        with col2:
            st.metric("📅 Date Range", date_range)
        with col3:
            st.metric("🛰️ Unique Tracks", f"{unique_tracks:,}")
        with col4:
            st.metric("📆 Observation Dates", observation_dates)
        
        st.info(f"📈 **Dataset Overview**: ITCC_v2_cleaned.csv with comprehensive meteorological data including Track_ID, Time_UTC, Area_km2, Tb_min, Tb_mean, CTH_mean_km, coordinates, velocity components, and lifecycle stages.")

# Standard filters
st.sidebar.header("🔍 Cloud Filters")

# Date filter
if 'timestamp' in df.columns:
    df["date"] = df["timestamp"].dt.date
    df["time_str"] = df["timestamp"].dt.strftime("%H:%M")
    available_dates = sorted(df["date"].dropna().unique())
    
    selected_date = st.sidebar.selectbox(
        "Select Date",
        available_dates,
        format_func=lambda d: d.strftime("%d %B %Y")
    )
    
    # Time filter
    available_times = sorted(
        df.loc[df["date"] == selected_date, "time_str"].dropna().unique(),
        key=lambda x: pd.to_datetime(x, format="%H:%M")
    )
    selected_time = st.sidebar.selectbox(
        "Select Time",
        available_times
    )
    # Filter by date and exact satellite capture time
    df_filtered = df[
        (df["date"] == selected_date) &
        (df["time_str"] == selected_time)
    ].copy()
else:
    df_filtered = df.copy()

# Lifecycle stage filter
if 'Lifecycle_Stage' in df_filtered.columns:
    stages = ['All'] + sorted(df_filtered['Lifecycle_Stage'].dropna().unique().tolist())
    selected_stage = st.sidebar.selectbox("🔄 Lifecycle Stage", stages)
    
    if selected_stage != 'All':
        df_filtered = df_filtered[df_filtered['Lifecycle_Stage'] == selected_stage]

# Temperature filter
if 'Tb_min' in df_filtered.columns:
    tb_range = st.sidebar.slider(
        "🌡️ Tb_min Range (K)",
        min_value=float(df['Tb_min'].min()),
        max_value=float(df['Tb_min'].max()),
        value=(float(df['Tb_min'].min()), float(df['Tb_min'].max())),
        step=1.0
    )
    
    df_filtered = df_filtered[
        (df_filtered['Tb_min'] >= tb_range[0]) &
        (df_filtered['Tb_min'] <= tb_range[1])
    ]

st.header("🌊 Cloud Map")
df_map = df_filtered
if "timestamp" in df.columns and "date" in df.columns:
    # For map animation, use all times for the selected date (otherwise a single selected_time leaves only 1 frame).
    df_map = df[df["date"] == selected_date].copy()
    if "Lifecycle_Stage" in df_map.columns and "Lifecycle_Stage" in df_filtered.columns:
        if "selected_stage" in locals() and selected_stage != "All":
            df_map = df_map[df_map["Lifecycle_Stage"] == selected_stage]
    if "Tb_min" in df_map.columns and "Tb_min" in df_filtered.columns:
        if "tb_range" in locals():
            df_map = df_map[(df_map["Tb_min"] >= tb_range[0]) & (df_map["Tb_min"] <= tb_range[1])]

if len(df_map) > 0:
    cloud_fig = create_animated_map(df_map)
    st.plotly_chart(
        cloud_fig,
        use_container_width=True,
        config={"scrollZoom": True, "displayModeBar": True}
    )
else:
    st.warning("No cloud clusters match the selected filters.")

st.header("🛰️ Trajectory")
if "Track_ID" in df.columns:
    # Let the user select a track based on the current filtered snapshot (date+time),
    # but always render the trajectory using the full lifecycle (unfiltered df).
    tracks_at_selected_time = (
        sorted(df_filtered["Track_ID"].dropna().unique()) if (len(df_filtered) > 0 and "Track_ID" in df_filtered.columns) else []
    )
    all_tracks = sorted(df["Track_ID"].dropna().unique())

    if tracks_at_selected_time:
        selected_track = st.selectbox(
            "Select Track",
            tracks_at_selected_time,
            format_func=lambda x: f"Track {x}"
        )
    elif all_tracks:
        selected_track = st.selectbox(
            "Select Track",
            all_tracks,
            format_func=lambda x: f"Track {x}"
        )
    else:
        selected_track = None
        st.warning("No tracks available in the dataset.")

    if selected_track is not None:
        df_track = df[df["Track_ID"] == selected_track].copy()
        df_track = df_track.dropna(subset=["Lat", "Lon"])

        if len(df_track) <= 1:
            st.warning("No trajectory data available for the selected track (need at least 2 observations).")
        else:
            trajectory_fig = create_animated_trajectory_map(df_track)
            st.plotly_chart(
                trajectory_fig,
                use_container_width=True,
                config={"scrollZoom": True, "displayModeBar": True}
            )

            df_track_sorted = df_track.copy()
            if "timestamp" not in df_track_sorted.columns and "Time_UTC" in df_track_sorted.columns:
                df_track_sorted["timestamp"] = pd.to_datetime(df_track_sorted["Time_UTC"], errors="coerce")
            if "timestamp" in df_track_sorted.columns:
                df_track_sorted["timestamp"] = pd.to_datetime(df_track_sorted["timestamp"], errors="coerce")
                df_track_sorted = df_track_sorted.dropna(subset=["timestamp"]).sort_values("timestamp")

            st.header("📊 Track Metrics")
            m1, m2, m3, m4 = st.columns(4)

            duration_label = "N/A"
            if "timestamp" in df_track_sorted.columns and len(df_track_sorted) >= 2:
                duration_hours = (df_track_sorted["timestamp"].max() - df_track_sorted["timestamp"].min()).total_seconds() / 3600.0
                duration_label = f"{duration_hours:.1f} hr"

            max_area_label = "N/A"
            if "Area_km2" in df_track_sorted.columns and len(df_track_sorted) > 0:
                max_area_label = f"{float(pd.to_numeric(df_track_sorted['Area_km2'], errors='coerce').max()):,.0f}"

            min_tb_label = "N/A"
            if "Tb_min" in df_track_sorted.columns and len(df_track_sorted) > 0:
                min_tb_label = f"{float(pd.to_numeric(df_track_sorted['Tb_min'], errors='coerce').min()):.1f}"

            mean_tb_label = "N/A"
            if "Tb_mean" in df_track_sorted.columns and len(df_track_sorted) > 0:
                mean_tb_label = f"{float(pd.to_numeric(df_track_sorted['Tb_mean'], errors='coerce').mean()):.1f}"
            elif "Tb_min" in df_track_sorted.columns and len(df_track_sorted) > 0:
                mean_tb_label = f"{float(pd.to_numeric(df_track_sorted['Tb_min'], errors='coerce').mean()):.1f}"

            with m1:
                st.metric("Duration", duration_label)
            with m2:
                st.metric("Max Area (km²)", max_area_label)
            with m3:
                st.metric("Min Tb (K)", min_tb_label)
            with m4:
                st.metric("Mean Tb (K)", mean_tb_label)

            st.header("📈 Track Evolution (Dual Axis)")
            if (
                "timestamp" in df_track_sorted.columns
                and "Area_km2" in df_track_sorted.columns
                and "Tb_min" in df_track_sorted.columns
                and len(df_track_sorted) > 1
            ):
                fig = create_dual_axis_track_evolution(df_track_sorted)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.warning("Track evolution unavailable (missing required columns or insufficient points).")

            st.header("🌪 Cloud Lifecycle Classification")
            stage_col = "Stage" if "Stage" in df_track_sorted.columns else ("Lifecycle_Stage" if "Lifecycle_Stage" in df_track_sorted.columns else None)
            if stage_col is not None and len(df_track_sorted) > 0:
                stage_counts = df_track_sorted[stage_col].dropna().value_counts().reset_index()
                stage_counts.columns = ["Stage", "Count"]

                fig_bar = px.bar(
                    stage_counts,
                    x="Stage",
                    y="Count",
                    color="Stage",
                    color_discrete_map={
                        "Developing": "green",
                        "Mature": "orange",
                        "Dissipating": "purple",
                    },
                )
                fig_bar.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#0a0e27",
                    paper_bgcolor="#0a0e27",
                    font=dict(color="#90caf9"),
                    height=350,
                    xaxis_title="Lifecycle Stage",
                    yaxis_title="Count",
                )
                fig_bar.update_xaxes(tickangle=45)
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
            else:
                st.warning("Lifecycle distribution unavailable (missing Stage/Lifecycle_Stage column).")
else:
    st.warning("No tracks available (missing Track_ID column).")



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

# Dark Theme CSS
st.markdown("""
<style>
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
    
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .stMetric {
        background-color: rgba(13, 27, 42, 0.9);
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
    
    h1, h2, h3, h4 {
        color: #90caf9 !important;
        font-weight: 600;
    }
    
    .stDataFrame {
        background-color: rgba(13, 27, 42, 0.7);
        border: 1px solid rgba(100, 181, 246, 0.2);
        border-radius: 8px;
    }
    
    /* Dark theme for plots */
    .js-plotly-plot .plotly {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 100%) !important;
    }
    
    .stSelectbox > div > div > select,
    .stSlider > div > div > div,
    .stDateInput > div > div > input {
        background-color: rgba(13, 27, 42, 0.8);
        color: #90caf9;
        border: 1px solid rgba(100, 181, 246, 0.3);
    }
    
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
    
    .animation-controls {
        background: rgba(13, 27, 42, 0.9);
        border: 1px solid rgba(100, 181, 246, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
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

# ============================================================================
# ANIMATION FUNCTIONS
# ============================================================================

def create_animated_map(df, animation_frame_col='frame_index'):
    """Create animated map with play/pause controls"""
    if len(df) == 0:
        return go.Figure()
    
    # Color based on temperature (dark theme)
    df['color'] = df['Tb_min'].apply(
        lambda x: '#ffffff' if x < 200 else '#b3e5fc' if x < 220 else '#4fc3f7'
    )
    
    # Size based on area
    df['size'] = np.clip(df['Area_km2'] / 80, 8, 40)
    
    fig = px.scatter_mapbox(
        df,
        lat='Lat',
        lon='Lon',
        color='color',
        size='size',
        animation_frame=animation_frame_col,
        hover_data=['Track_ID', 'Tb_min', 'Area_km2', 'CTH_max_km', 'Lifecycle_Stage'],
        zoom=4,
        center={'lat': 12.5, 'lon': 75},
        mapbox_style='carto-darkmatter',
        title='🌊 Animated Cloud Cluster Movement - Indian Ocean',
        color_discrete_map={
            '#ffffff': 'Coldest',
            '#b3e5fc': 'Medium', 
            '#4fc3f7': 'Warm'
        }
    )
    
    # Animation settings
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['redraw'] = True
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300
    
    fig.update_layout(
        height=650,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        showlegend=True,
        font=dict(color='#90caf9', size=12),
        legend=dict(
            bgcolor='rgba(13, 27, 42, 0.8)',
            bordercolor='rgba(100, 181, 246, 0.3)',
            borderwidth=1
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.011,
                xanchor="left",
                y=0,
                yanchor="top",
                buttons=[
                    dict(
                        label="▶️ Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 800, "redraw": True},
                                     "fromcurrent": True, "transition": {"duration": 300}}]
                    ),
                    dict(
                        label="⏸️ Pause", 
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate", "transition": {"duration": 0}}]
                    )
                ]
            )
        ]
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate='<b>Track %{customdata[0]}</b><br>' +
                     'Lat: %{lat:.2f}°<br>' +
                     'Lon: %{lon:.2f}°<br>' +
                     'Tb_min: %{customdata[1]:.1f} K<br>' +
                     'Area: %{customdata[2]:.0f} km²<br>' +
                     'CTH: %{customdata[3]:.1f} km<br>' +
                     'Stage: %{customdata[4]}<extra></extra>',
        customdata=df[['Track_ID', 'Tb_min', 'Area_km2', 'CTH_max_km', 'Lifecycle_Stage']]
    )
    
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
            name='Area (km²)',
            hovertemplate='Time: %{x}<br>Area: %{y:.0f} km²<br>Frame: %{marker.color}<extra></extra>'
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
            hovertemplate='Time: %{x}<br>Tb_min: %{y:.1f} K<br>Frame: %{marker.color}<extra></extra>'
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
    fig.update_yaxes(title_text="Area (km²)", row=1, col=1)
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

# Animation controls in sidebar
st.sidebar.header("🎬 Animation Controls")

# Animation mode toggle
animation_mode = st.sidebar.checkbox("🎥 Enable Animation Mode", value=True)

if animation_mode:
    st.sidebar.markdown('<div class="animation-controls">', unsafe_allow_html=True)
    
    # Animation speed control
    animation_speed = st.sidebar.slider(
        "⚡ Animation Speed",
        min_value=200,
        max_value=2000,
        value=800,
        step=100,
        help="Frame duration in milliseconds"
    )
    
    # Auto-play option
    auto_play = st.sidebar.checkbox("🔄 Auto-play on load")
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")

# Standard filters
st.sidebar.header("🔍 Cloud Filters")

# Date filter
if 'timestamp' in df.columns:
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    selected_date = st.sidebar.date_input(
        "📅 Select Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Hour filter
    available_hours = sorted(df['timestamp'].dt.hour.unique())
    selected_hour = st.sidebar.selectbox("⏰ Select Hour (UTC)", available_hours)
    
    # Filter by date and hour
    df_filtered = df[
        (df['timestamp'].dt.date == selected_date) &
        (df['timestamp'].dt.hour == selected_hour)
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

# Quality filter
if 'quality_flag' in df_filtered.columns:
    quality_options = st.sidebar.multiselect(
        "✅ Quality Flag",
        options=[0, 1, 2],
        default=[0],
        format_func=lambda x: {0: "Good", 1: "Suspect", 2: "Missing"}[x]
    )
    
    if quality_options:
        df_filtered = df_filtered[df_filtered['quality_flag'].isin(quality_options)]

# Summary metrics with dark theme
st.header("📊 Ocean Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🌊 Total Observations", f"{len(df):,}")

with col2:
    if 'Track_ID' in df.columns:
        unique_tracks = df['Track_ID'].nunique()
        st.metric("🛰️ Unique Tracks", f"{unique_tracks:,}")
    else:
        st.metric("🛰️ Unique Tracks", "N/A")

with col3:
    st.metric("👁️ Currently Displayed", f"{len(df_filtered)}")

with col4:
    if 'timestamp' in df.columns:
        duration = (df['timestamp'].max() - df['timestamp'].min()).days
        st.metric("📅 Dataset Duration", f"{duration} days")
    else:
        st.metric("📅 Dataset Duration", "N/A")

# Animation status indicator
if animation_mode:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; margin: 1rem 0;">
            <span class="status-indicator status-paused"></span>
            <span style="color: #90caf9; font-weight: 600;">
                Animation: Ready | Speed: {animation_speed}ms/frame
            </span>
        </div>
        """, unsafe_allow_html=True)

# Main animated map
st.header("🗺️ Animated Cloud Movement")
if len(df_filtered) > 0:
    if animation_mode:
        # Create animated version
        animated_fig = create_animated_map(df_filtered)
        st.plotly_chart(animated_fig, use_container_width=True)
        
        # Animation instructions
        st.info("🎬 **Animation Controls**: Use the play/pause buttons below the map to control animation. You can also drag the slider to jump to specific frames.")
    else:
        # Static version
        static_fig = create_animated_map(df_filtered, animation_frame_col=None)
        st.plotly_chart(static_fig, use_container_width=True)
    
    # Lifecycle distribution
    if 'Lifecycle_Stage' in df_filtered.columns:
        st.subheader("🔄 Lifecycle Distribution")
        stage_counts = df_filtered['Lifecycle_Stage'].value_counts()
        cols = st.columns(min(len(stage_counts), 4))
        
        for i, (stage, count) in enumerate(stage_counts.items()):
            if i < len(cols):
                with cols[i]:
                    st.metric(stage, int(count))
else:
    st.warning("🌊 No cloud clusters match current filters")

# Track analysis with animation support
if 'Track_ID' in df.columns:
    st.header("🛰️ Track Analysis with Animation")
    
    available_tracks = sorted(df['Track_ID'].unique())
    selected_track = st.selectbox(
        "Select Track ID for Animation",
        options=available_tracks,
        format_func=lambda x: f"Track {x}"
    )
    
    if selected_track:
        df_track = df[df['Track_ID'] == selected_track].copy()
        
        if len(df_track) > 0:
            # Track summary
            st.subheader(f"📊 Track {selected_track} Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'timestamp' in df_track.columns:
                    duration_hours = (df_track['timestamp'].max() - df_track['timestamp'].min()).total_seconds() / 3600
                    st.metric("⏱️ Duration", f"{duration_hours:.1f} hrs")
                else:
                    st.metric("⏱️ Duration", f"{len(df_track)} obs")
            
            with col2:
                st.metric("📏 Max Area", f"{df_track['Area_km2'].max():.0f} km²")
            
            with col3:
                st.metric("🌡️ Min Tb_min", f"{df_track['Tb_min'].min():.1f} K")
            
            with col4:
                st.metric("☁️ Max CTH", f"{df_track['CTH_max_km'].max():.1f} km")
            
            # Animated time series
            st.subheader("📈 Animated Evolution")
            time_series_fig = create_time_series(df_track)
            st.plotly_chart(time_series_fig, use_container_width=True)
            
            # Track data table with frame info
            with st.expander("📋 View Track Data (Animation Frames)"):
                display_cols = ['Time_UTC', 'Lat', 'Lon', 'Area_km2', 'Tb_min', 'Lifecycle_Stage', 'frame_index']
                available_cols = [col for col in display_cols if col in df_track.columns]
                st.dataframe(
                    df_track[available_cols].sort_values('Time_UTC'),
                    use_container_width=True,
                    height=300
                )

# Footer with dark theme
st.markdown("""
<div class="dark-footer">
    <strong>🌊 ITCC Animated Dashboard</strong> | INSAT-3D Satellite Data | Time-lapse Animation | Dark Theme Interface
</div>
""", unsafe_allow_html=True)
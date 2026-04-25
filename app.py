import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import os

# Set page config
st.set_page_config(
    page_title="Tropical Cloud Cluster Tracking Dashboard",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stDataFrame {
        background-color: white;
    }
    
    /* Dark theme for plots */
    .js-plotly-plot .plotly {
        background-color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

# Load sample data
@st.cache_data
def load_sample_data():
    """Load sample TCC data for demonstration"""
    # Create sample data that mimics real TCC observations
    np.random.seed(42)
    
    # Generate sample tracks
    tracks = []
    base_date = datetime(2024, 1, 1)
    
    for track_id in range(1, 11):  # 10 sample tracks
        # Generate 20-30 observations per track
        n_obs = np.random.randint(20, 31)
        
        for i in range(n_obs):
            # Time progression
            time_UTC = base_date + timedelta(hours=i*0.5)
            
            # Spatial movement (simulate cloud movement)
            lat = 10.0 + track_id * 2 + np.random.normal(0, 0.5)
            lon = 70.0 + track_id * 3 + i * 0.2 + np.random.normal(0, 0.3)
            
            # Cloud properties
            area_km2 = np.random.uniform(500, 5000)
            tb_min = np.random.uniform(200, 240)
            tb_mean = tb_min + np.random.uniform(10, 30)
            cth_max_km = np.random.uniform(8, 15)
            
            # Lifecycle stage
            if i < n_obs * 0.3:
                stage = 'Developing'
            elif i < n_obs * 0.7:
                stage = 'Mature'
            else:
                stage = 'Dissipating'
            
            tracks.append({
                'Track_ID': track_id,
                'Time_UTC': time_UTC,
                'Date': time_UTC.date(),
                'Hour': time_UTC.hour,
                'Lat': lat,
                'Lon': lon,
                'Area_km2': area_km2,
                'Tb_min': tb_min,
                'Tb_mean': tb_mean,
                'CTH_max_km': cth_max_km,
                'Lifecycle_Stage': stage
            })
    
    return pd.DataFrame(tracks)

# Load data
with st.spinner("Loading Tropical Cloud Cluster data..."):
    df = load_sample_data()

# Header
st.markdown('<div class="main-header">☁️ Tropical Cloud Cluster Tracking Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">INSAT-3D Satellite Data Analysis & Visualization</div>', unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Date selector
available_dates = sorted(df['Date'].unique())
selected_date = st.sidebar.selectbox(
    "Select Date",
    options=available_dates,
    format_func=lambda x: x.strftime('%Y-%m-%d')
)

# Hour selector
available_hours = sorted(df['Hour'].unique())
selected_hour = st.sidebar.selectbox("Select Hour", options=available_hours)

# Lifecycle stage filter
stages = ['All'] + sorted(df['Lifecycle_Stage'].dropna().unique().tolist())
selected_stage = st.sidebar.selectbox("Lifecycle Stage", stages)

# Tb_min range filter
tb_min_range = st.sidebar.slider(
    "Tb_min Range (K)",
    min_value=float(df['Tb_min'].min()),
    max_value=float(df['Tb_min'].max()),
    value=(float(df['Tb_min'].min()), float(df['Tb_min'].max())),
    step=1.0
)

# Apply filters
df_filtered = df[
    (df['Date'] == selected_date) &
    (df['Hour'] == selected_hour)
].copy()

if selected_stage != 'All':
    df_filtered = df_filtered[df_filtered['Lifecycle_Stage'] == selected_stage]

df_filtered = df_filtered[
    (df_filtered['Tb_min'] >= tb_min_range[0]) &
    (df_filtered['Tb_min'] <= tb_min_range[1])
]

# Summary metrics
st.header("📊 Summary Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Observations", f"{len(df):,}")

with col2:
    st.metric("Unique Tracks", f"{df['Track_ID'].nunique()}")

with col3:
    st.metric("Current Display", f"{len(df_filtered)}")

with col4:
    duration = (df['Date'].max() - df['Date'].min()).days
    st.metric("Dataset Duration", f"{duration} days")

# Map visualization
st.header("🗺️ Cloud Cluster Locations")

if len(df_filtered) > 0:
    # Create map
    fig = go.Figure()
    
    # Add cloud clusters
    stage_colors = {
        'Developing': '#10b981',
        'Mature': '#f59e0b', 
        'Dissipating': '#ef4444'
    }
    
    for stage, color in stage_colors.items():
        stage_data = df_filtered[df_filtered['Lifecycle_Stage'] == stage]
        if len(stage_data) > 0:
            fig.add_trace(go.Scattermapbox(
                lat=stage_data['Lat'],
                lon=stage_data['Lon'],
                mode='markers',
                marker=dict(
                    size=stage_data['Area_km2'] / 500,  # Size based on area
                    color=color,
                    opacity=0.7
                ),
                name=stage,
                text=[f"Track: {tid}<br>Area: {area:.0f} km²<br>Tb_min: {tb:.1f} K"
                      for tid, area, tb in zip(stage_data['Track_ID'], 
                                             stage_data['Area_km2'], 
                                             stage_data['Tb_min'])],
                hovertemplate='%{text}<extra></extra>'
            ))
    
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            zoom=5,
            center=dict(lat=df_filtered['Lat'].mean(), lon=df_filtered['Lon'].mean())
        ),
        height=500,
        title=f"TCCs on {selected_date.strftime('%Y-%m-%d')} at {selected_hour:02d}:00 UTC",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Lifecycle distribution
    st.subheader("🔄 Lifecycle Distribution")
    stage_counts = df_filtered['Lifecycle_Stage'].value_counts()
    cols = st.columns(min(len(stage_counts), 4))
    
    for i, (stage, count) in enumerate(stage_counts.items()):
        if i < len(cols):
            with cols[i]:
                st.metric(stage, int(count))
else:
    st.warning("No cloud clusters match the current filter criteria")

# Track analysis
st.header("🔍 Track-Level Analysis")

# Track selector
available_tracks = sorted(df['Track_ID'].unique())
selected_track = st.selectbox(
    "Select Track ID",
    options=available_tracks,
    format_func=lambda x: f"Track {x}"
)

if selected_track:
    df_track = df[df['Track_ID'] == selected_track].copy()
    
    if len(df_track) > 0:
        # Track summary
        st.subheader(f"Track {selected_track} Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            duration_hours = (df_track['Time_UTC'].max() - df_track['Time_UTC'].min()).total_seconds() / 3600
            st.metric("Duration", f"{duration_hours:.1f} hrs")
        
        with col2:
            st.metric("Max Area", f"{df_track['Area_km2'].max():.0f} km²")
        
        with col3:
            st.metric("Min Tb_min", f"{df_track['Tb_min'].min():.1f} K")
        
        with col4:
            st.metric("Max CTH", f"{df_track['CTH_max_km'].max():.1f} km")
        
        # Evolution graph
        st.subheader("📈 Track Evolution")
        
        df_track_sorted = df_track.sort_values('Time_UTC')
        
        # Create dual-axis plot
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]]
        )
        
        # Area line
        fig.add_trace(
            go.Scatter(
                x=df_track_sorted['Time_UTC'],
                y=df_track_sorted['Area_km2'],
                mode='lines',
                line=dict(color='#00CED1', width=3),
                name='Area (km²)',
                hovertemplate='Time: %{x}<br>Area: %{y:.0f} km²<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Tb_min line
        fig.add_trace(
            go.Scatter(
                x=df_track_sorted['Time_UTC'],
                y=df_track_sorted['Tb_min'],
                mode='lines',
                line=dict(color='#DC143C', width=2, dash='dot'),
                name='Tb_min (K)',
                hovertemplate='Time: %{x}<br>Tb_min: %{y:.1f} K<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Time (UTC)")
        fig.update_yaxes(title_text="Area (km²)", secondary_y=False)
        fig.update_yaxes(title_text="Tb_min (K)", secondary_y=True)
        
        fig.update_layout(
            title=f"Track {selected_track} Evolution",
            height=400,
            plot_bgcolor='#1e293b',
            paper_bgcolor='#1e293b',
            font=dict(color='#ffffff')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Track data table
        with st.expander("📋 View Track Data"):
            display_cols = ['Time_UTC', 'Lat', 'Lon', 'Area_km2', 'Tb_min', 'Lifecycle_Stage']
            st.dataframe(
                df_track[display_cols].sort_values('Time_UTC'),
                use_container_width=True,
                height=300
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p><b>Tropical Cloud Cluster Tracking Dashboard</b></p>
    <p>INSAT-3D Satellite Data Analysis System</p>
    <p>Real-time monitoring of tropical convective systems</p>
</div>
""", unsafe_allow_html=True)

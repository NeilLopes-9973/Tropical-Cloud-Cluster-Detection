"""
ITCC Interactive Dashboard
==========================
Interactive Streamlit dashboard for visualizing Tropical Cloud Clusters (TCCs)
from INSAT-3D satellite data with temporal and spatial filtering.

Features:
- Interactive map with time filtering
- Lifecycle stage classification
- Track-level analysis
- Animated time-lapse
- Dual-axis time series plots

Author: Neil Lopes
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="ITCC Dashboard",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0e1117;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
        border: 1px solid #475569;
    }
    
    .stMetric label {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="metric-container"] * {
        color: #ffffff !important;
    }
    
    .sidebar .sidebar-content {
        background-color: #1e293b;
    }
    
    /* Fix sidebar text colors */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #ffffff !important;
    }
    
    /* Fix date picker and dropdown text */
    [data-testid="stSidebar"] input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] select {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stDateInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * {
        color: #000000 !important;
    }
    
    /* Force all input fields to have dark text */
    section[data-testid="stSidebar"] input[type="text"],
    section[data-testid="stSidebar"] input[type="date"] {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Fix filter panel text */
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.6);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING AND CACHING
# ============================================================================

@st.cache_data(show_spinner=False)
def load_data(filepath='../data/final/ITCC_v2_cleaned.csv'):
    """
    Load and preprocess ITCC v2 dataset.
    
    Args:
        filepath: Path to ITCC_v2.csv file
    
    Returns:
        DataFrame with parsed timestamps and sorted by time
    """
    try:
        df = pd.read_csv(filepath)
        
        # Parse timestamp
        df['Time_UTC'] = pd.to_datetime(df['Time_UTC'])
        
        # Fix column names - use lowercase lat/lon which have correct values
        if 'lat' in df.columns and 'lon' in df.columns:
            df['Lat'] = df['lat']
            df['Lon'] = df['lon']
        
        # Extract date and hour for filtering
        df['Date'] = df['Time_UTC'].dt.date
        df['Hour'] = df['Time_UTC'].dt.hour
        df['Minute'] = df['Time_UTC'].dt.minute
        
        # Sort by time
        df = df.sort_values('Time_UTC').reset_index(drop=True)
        
        # Ensure numeric columns
        numeric_cols = ['Lat', 'Lon', 'Area_km2', 'Tb_min', 'Tb_mean', 'CTH_max_km']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    except FileNotFoundError:
        st.error(f"Error: File not found at {filepath}")
        st.info("Please ensure ITCC_v2.csv exists in data/final/ directory")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def get_dataset_stats(df):
    """
    Compute summary statistics for the dataset.
    
    Args:
        df: ITCC DataFrame
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_observations': len(df),
        'unique_tracks': df['Track_ID'].nunique(),
        'date_range': (df['Date'].min(), df['Date'].max()),
        'time_range': (df['Time_UTC'].min(), df['Time_UTC'].max()),
        'spatial_extent': {
            'lat_min': df['Lat'].min(),
            'lat_max': df['Lat'].max(),
            'lon_min': df['Lon'].min(),
            'lon_max': df['Lon'].max()
        }
    }
    
    if 'Lifecycle_Stage' in df.columns:
        stats['lifecycle_counts'] = df['Lifecycle_Stage'].value_counts().to_dict()
    
    return stats


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_map_visualization(df_filtered, title="TCC Locations"):
    """
    Create interactive map with TCC locations.
    
    Args:
        df_filtered: Filtered DataFrame
        title: Map title
    
    Returns:
        Plotly figure
    """
    if len(df_filtered) == 0:
        # Empty map
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(height=600)
        return fig
    
    # Normalize area for marker size (10-50 range - BIGGER markers)
    area_normalized = df_filtered['Area_km2'].fillna(0)
    size_min, size_max = 10, 50
    if area_normalized.max() > area_normalized.min():
        sizes = size_min + (area_normalized - area_normalized.min()) / \
                (area_normalized.max() - area_normalized.min()) * (size_max - size_min)
    else:
        sizes = [25] * len(df_filtered)
    
    # Create hover text
    hover_text = []
    for _, row in df_filtered.iterrows():
        text = f"<b>Track {row['Track_ID']}</b><br>"
        text += f"Time: {row['Time_UTC']}<br>"
        text += f"Tb_min: {row['Tb_min']:.1f} K<br>"
        text += f"Area: {row['Area_km2']:.0f} km²<br>"
        text += f"CTH: {row.get('CTH_max_km', 0):.1f} km<br>"
        text += f"Stage: {row.get('Lifecycle_Stage', 'N/A')}"
        hover_text.append(text)
    
    # Create map
    fig = px.scatter_mapbox(
        df_filtered,
        lat='Lat',
        lon='Lon',
        color='Tb_min',
        size=sizes,
        hover_name='Track_ID',
        hover_data={
            'Lat': ':.2f',
            'Lon': ':.2f',
            'Tb_min': ':.1f',
            'Area_km2': ':.0f',
            'CTH_max_km': ':.1f'
        },
        color_continuous_scale=[[0.0, "#FFFFFF"], [0.5, "#8ECFFF"], [1.0, "#3B82F6"]],  # White to blue (cold to warm)
        mapbox_style='carto-darkmatter',
        zoom=3,
        center={'lat': df_filtered['Lat'].mean(), 'lon': df_filtered['Lon'].mean()},
        title=title,
        labels={
            'Tb_min': 'Brightness Temp (K)',
            'Area_km2': 'Area (km²)',
            'CTH_max_km': 'Cloud Top Height (km)'
        }
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        title_font_size=18,
        title_font_color='#1f77b4',
        coloraxis_colorbar=dict(
            title="Tb (K)",
            thickness=15,
            len=0.7
        )
    )
    
    # Update marker appearance - keep default hover with all data
    fig.update_traces(
        marker=dict(
            opacity=0.7
        )
    )
    
    return fig


def create_animated_map(df_filtered):
    """
    Create animated time-lapse map.
    
    Args:
        df_filtered: Filtered DataFrame
    
    Returns:
        Plotly figure with animation
    """
    if len(df_filtered) == 0:
        return create_map_visualization(df_filtered, "No data for animation")
    
    # Normalize area for marker size (10-50 range - BIGGER markers)
    area_normalized = df_filtered['Area_km2'].fillna(0)
    size_min, size_max = 10, 50
    if area_normalized.max() > area_normalized.min():
        sizes = size_min + (area_normalized - area_normalized.min()) / \
                (area_normalized.max() - area_normalized.min()) * (size_max - size_min)
    else:
        sizes = [25] * len(df_filtered)
    
    df_filtered = df_filtered.copy()
    df_filtered['size'] = sizes
    
    # Create animated map
    fig = px.scatter_mapbox(
        df_filtered,
        lat='Lat',
        lon='Lon',
        color='Tb_min',
        size='size',
        animation_frame='Time_UTC',
        hover_name='Track_ID',
        hover_data={
            'Lat': ':.2f',
            'Lon': ':.2f',
            'Tb_min': ':.1f',
            'Area_km2': ':.0f',
            'size': False
        },
        color_continuous_scale=[[0.0, "#FFFFFF"], [0.5, "#8ECFFF"], [1.0, "#3B82F6"]],
        mapbox_style='carto-darkmatter',
        zoom=3,
        center={'lat': df_filtered['Lat'].mean(), 'lon': df_filtered['Lon'].mean()},
        title="TCC Time-Lapse Animation",
        labels={'Tb_min': 'Brightness Temp (K)'}
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        title_font_size=18,
        title_font_color='#1f77b4',
        coloraxis_colorbar=dict(
            title="Tb (K)",
            thickness=15,
            len=0.7
        )
    )
    
    # Update animation settings
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 500
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300
    
    return fig


def create_track_trajectory_map(df_track):
    """
    Create trajectory map for a single track.
    
    Args:
        df_track: DataFrame for single track
    
    Returns:
        Plotly figure
    """
    if len(df_track) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data for selected track",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Sort by time
    df_track = df_track.sort_values('Time_UTC')
    
    # Create time numeric for coloring
    df_track['time_numeric'] = (df_track['Time_UTC'] - df_track['Time_UTC'].min()).dt.total_seconds() / 3600
    
    # Create figure
    fig = go.Figure()
    
    # Add trajectory line
    fig.add_trace(go.Scattermapbox(
        lat=df_track['Lat'],
        lon=df_track['Lon'],
        mode='lines+markers',
        marker=dict(
            size=8,
            color=df_track['time_numeric'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Hours", x=1.1)
        ),
        text=[f"Time: {t}<br>Tb: {tb:.1f} K" 
              for t, tb in zip(df_track['Time_UTC'], df_track['Tb_min'])],
        hovertemplate='%{text}<extra></extra>',
        name='Track'
    ))
    
    # Add start and end markers
    fig.add_trace(go.Scattermapbox(
        lat=[df_track.iloc[0]['Lat']],
        lon=[df_track.iloc[0]['Lon']],
        mode='markers',
        marker=dict(size=15, color='green', symbol='circle'),
        text=['Start'],
        hovertemplate='<b>Start</b><br>%{lat:.2f}, %{lon:.2f}<extra></extra>',
        name='Start'
    ))
    
    fig.add_trace(go.Scattermapbox(
        lat=[df_track.iloc[-1]['Lat']],
        lon=[df_track.iloc[-1]['Lon']],
        mode='markers',
        marker=dict(size=15, color='red', symbol='circle'),
        text=['End'],
        hovertemplate='<b>End</b><br>%{lat:.2f}, %{lon:.2f}<extra></extra>',
        name='End'
    ))
    
    # Update layout
    fig.update_layout(
        mapbox=dict(
            style='carto-positron',
            zoom=4,
            center=dict(
                lat=df_track['Lat'].mean(),
                lon=df_track['Lon'].mean()
            )
        ),
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"Track {df_track['Track_ID'].iloc[0]} Trajectory",
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig


def create_track_timeseries(df_track):
    """
    Create dual-axis time series for track evolution.
    
    Args:
        df_track: DataFrame for single track
    
    Returns:
        Plotly figure
    """
    if len(df_track) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data for selected track",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Sort by time
    df_track = df_track.sort_values('Time_UTC')
    
    # Create subplots with secondary y-axis
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    # Add Area trace
    fig.add_trace(
        go.Bar(
            x=df_track['Time_UTC'],
            y=df_track['Area_km2'],
            name='Area',
            marker_color='lightblue',
            opacity=0.7,
            hovertemplate='%{x}<br>Area: %{y:.0f} km²<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add Tb_min trace
    fig.add_trace(
        go.Scatter(
            x=df_track['Time_UTC'],
            y=df_track['Tb_min'],
            name='Tb_min',
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            hovertemplate='%{x}<br>Tb_min: %{y:.1f} K<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Add lifecycle stage markers if available
    if 'Lifecycle_Stage' in df_track.columns:
        stage_colors = {
            'Developing': 'green',
            'Mature': 'orange',
            'Dissipating': 'purple'
        }
        
        for stage, color in stage_colors.items():
            stage_data = df_track[df_track['Lifecycle_Stage'] == stage]
            if len(stage_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=stage_data['Time_UTC'],
                        y=stage_data['Area_km2'],
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=color,
                            symbol='diamond',
                            line=dict(width=1, color='white')
                        ),
                        name=stage,
                        hovertemplate=f'<b>{stage}</b><br>%{{x}}<extra></extra>'
                    ),
                    secondary_y=False
                )
    
    # Update axes
    fig.update_xaxes(title_text="Time (UTC)", showgrid=True)
    fig.update_yaxes(title_text="<b>Area (km²)</b>", secondary_y=False, showgrid=True)
    fig.update_yaxes(title_text="<b>Tb_min (K)</b>", secondary_y=True, showgrid=False)
    
    # Update layout
    fig.update_layout(
        height=400,
        title=f"Track {df_track['Track_ID'].iloc[0]} Evolution",
        title_font_size=16,
        title_font_color='#1f77b4',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    
    return fig


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main dashboard application."""
    
    # Load data
    with st.spinner("Loading ITCC v2 dataset..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check if ITCC_v2.csv exists in data/final/ directory.")
        st.info("Run: cd 01_data_preparation && python enhance_itcc_dataset.py")
        st.stop()
    
    # Get statistics
    stats = get_dataset_stats(df)
    
    # Show data info
    with st.expander("ℹ️ Dataset Information", expanded=False):
        st.write(f"**Total Records**: {len(df):,}")
        st.write(f"**Date Range**: {df['Date'].min()} to {df['Date'].max()}")
        st.write(f"**Unique Tracks**: {df['Track_ID'].nunique()}")
        st.write(f"**Available Dates**: {sorted(df['Date'].unique())}")
    
    # ========================================================================
    # SIDEBAR - FILTERS
    # ========================================================================
    
    st.sidebar.header("🔍 Filters")
    
    # Date selector
    st.sidebar.subheader("📅 Date Selection")
    available_dates = sorted(df['Date'].unique())
    
    selected_date = st.sidebar.selectbox(
        "Select Date",
        options=available_dates,
        format_func=lambda x: x.strftime('%Y-%m-%d')
    )
    
    # Hour slider
    st.sidebar.subheader("🕐 Time Selection")
    selected_hour = st.sidebar.slider(
        "Hour (UTC)",
        min_value=0,
        max_value=23,
        value=12,
        step=1,
        format="%d:00"
    )
    
    # Lifecycle stage filter
    st.sidebar.subheader("🔄 Lifecycle Stage")
    if 'Lifecycle_Stage' in df.columns:
        stages = ['All'] + sorted(df['Lifecycle_Stage'].dropna().unique().tolist())
        selected_stage = st.sidebar.selectbox("Filter by Stage", stages)
    else:
        selected_stage = 'All'
    
    # Tb_min filter
    st.sidebar.subheader("🌡️ Convective Intensity")
    tb_min_range = st.sidebar.slider(
        "Tb_min Range (K)",
        min_value=float(df['Tb_min'].min()),
        max_value=float(df['Tb_min'].max()),
        value=(float(df['Tb_min'].min()), float(df['Tb_min'].max())),
        step=1.0
    )
    
    # Animation toggle
    st.sidebar.subheader("🎬 Animation")
    show_animation = st.sidebar.checkbox("Play Animation", value=False)
    
    # ========================================================================
    # APPLY FILTERS
    # ========================================================================
    
    # Filter by date and hour (show all data within selected hour)
    df_filtered = df[
        (df['Date'] == selected_date) &
        (df['Hour'] == selected_hour)
    ].copy()
    
    # Debug: Show filter info
    st.sidebar.info(f"Filtered: {len(df_filtered)} TCCs at {selected_date} {selected_hour:02d}:00")
    
    # Filter by lifecycle stage
    if selected_stage != 'All' and 'Lifecycle_Stage' in df.columns:
        df_filtered = df_filtered[df_filtered['Lifecycle_Stage'] == selected_stage]
    
    # Filter by Tb_min
    df_filtered = df_filtered[
        (df_filtered['Tb_min'] >= tb_min_range[0]) &
        (df_filtered['Tb_min'] <= tb_min_range[1])
    ]
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    st.markdown(
        '<div class="main-header">☁️ INSAT-3D Tropical Cloud Cluster Tracking Dashboard</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">AI/ML-based detection and tracking of tropical cloud clusters using INSAT-3D IRBRT data</div>',
        unsafe_allow_html=True
    )
    
    # ========================================================================
    # MAIN CONTENT - SUMMARY METRICS
    # ========================================================================
    
    st.header("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Observations",
            value=f"{stats['total_observations']:,}"
        )
    
    with col2:
        st.metric(
            label="Unique Tracks",
            value=f"{stats['unique_tracks']:,}"
        )
    
    with col3:
        st.metric(
            label="Currently Displayed",
            value=f"{len(df_filtered):,}"
        )
    
    with col4:
        duration = (stats['time_range'][1] - stats['time_range'][0]).days
        st.metric(
            label="Dataset Duration",
            value=f"{duration} days"
        )
    
    # ========================================================================
    # MAIN MAP VISUALIZATION
    # ========================================================================
    
    st.header("🗺️ TCC Locations")
    
    if show_animation:
        st.info("🎬 Animation Mode: Showing time-lapse for selected date")
        
        # Filter for entire selected date
        df_animation = df[df['Date'] == selected_date].copy()
        
        # Apply other filters
        if selected_stage != 'All' and 'Lifecycle_Stage' in df.columns:
            df_animation = df_animation[df_animation['Lifecycle_Stage'] == selected_stage]
        
        df_animation = df_animation[
            (df_animation['Tb_min'] >= tb_min_range[0]) &
            (df_animation['Tb_min'] <= tb_min_range[1])
        ]
        
        fig_map = create_animated_map(df_animation)
    else:
        title = f"TCCs on {selected_date.strftime('%Y-%m-%d')} at {selected_hour:02d}:00 UTC"
        fig_map = create_map_visualization(df_filtered, title)
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Display summary
    if len(df_filtered) > 0:
        st.success(f"✅ Displaying {len(df_filtered)} TCC(s)")
        
        # Lifecycle breakdown
        if 'Lifecycle_Stage' in df_filtered.columns:
            stage_counts = df_filtered['Lifecycle_Stage'].value_counts()
            if len(stage_counts) > 0:
                st.write("**Lifecycle Distribution:**")
                num_cols = min(len(stage_counts), 4)  # Max 4 columns
                cols = st.columns(num_cols)
                for i, (stage, count) in enumerate(stage_counts.items()):
                    if i < num_cols:
                        with cols[i]:
                            st.metric(label=str(stage), value=int(count))
    else:
        st.warning("⚠️ No TCCs match the current filter criteria")
    
    # ========================================================================
    # CYCLOGENESIS RISK ASSESSMENT
    # ========================================================================
    
    st.markdown("---")
    st.header("🌪️ Cyclogenesis Risk Assessment")
    
    # Simulate cyclogenesis probability (replace with actual model predictions)
    if len(df_filtered) > 0:
        # Calculate risk based on TCC characteristics
        df_risk = df_filtered.copy()
        
        # Simple risk scoring based on physical parameters
        # (Replace this with actual ML model predictions)
        risk_score = (
            (df_risk['Tb_min'] < 210).astype(int) * 30 +  # Very cold clouds
            (df_risk['Area_km2'] > 10000).astype(int) * 25 +  # Large area
            (df_risk.get('CTH_max_km', 0) > 12).astype(int) * 20 +  # High cloud tops
            (df_risk['Lifecycle_Stage'] == 'Mature').astype(int) * 25  # Mature stage
        )
        
        df_risk['Cyclogenesis_Probability'] = np.clip(risk_score, 0, 100)
        df_risk['Risk_Level'] = pd.cut(
            df_risk['Cyclogenesis_Probability'],
            bins=[0, 30, 60, 100],
            labels=['Low', 'Medium', 'High']
        )
        
        # Risk summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            high_risk = (df_risk['Risk_Level'] == 'High').sum()
            st.metric("High Risk Clusters", high_risk, 
                     delta="⚠️" if high_risk > 0 else "✓",
                     delta_color="inverse")
        
        with col2:
            medium_risk = (df_risk['Risk_Level'] == 'Medium').sum()
            st.metric("Medium Risk Clusters", medium_risk)
        
        with col3:
            low_risk = (df_risk['Risk_Level'] == 'Low').sum()
            st.metric("Low Risk Clusters", low_risk)
        
        with col4:
            avg_prob = df_risk['Cyclogenesis_Probability'].mean()
            st.metric("Avg Probability", f"{avg_prob:.1f}%")
        
        # High-risk clusters table
        high_risk_clusters = df_risk[df_risk['Risk_Level'] == 'High'].sort_values(
            'Cyclogenesis_Probability', ascending=False
        )
        
        if len(high_risk_clusters) > 0:
            st.subheader("⚠️ High-Risk Clusters Requiring Attention")
            
            display_cols = ['Track_ID', 'Time_UTC', 'Lat', 'Lon', 'Tb_min', 
                           'Area_km2', 'CTH_max_km', 'Cyclogenesis_Probability', 'Risk_Level']
            display_cols = [col for col in display_cols if col in high_risk_clusters.columns]
            
            st.dataframe(
                high_risk_clusters[display_cols].head(10),
                use_container_width=True,
                height=300
            )
            
            st.warning("⚠️ **Alert**: High cyclogenesis probability detected. Monitor these clusters closely.")
        else:
            st.info("✓ No high-risk clusters detected in current selection.")
        
        # Risk distribution chart
        with st.expander("📊 View Risk Distribution"):
            risk_counts = df_risk['Risk_Level'].value_counts()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    marker_color=['#10b981', '#f59e0b', '#ef4444'],
                    text=risk_counts.values,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Cyclogenesis Risk Distribution",
                xaxis_title="Risk Level",
                yaxis_title="Number of Clusters",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for cyclogenesis assessment.")
    
    # ========================================================================
    # TRACK-LEVEL DETAIL VIEW
    # ========================================================================
    
    st.header("🔍 Track-Level Analysis")
    
    # Track selector
    try:
        available_tracks = sorted(df['Track_ID'].unique())
    except Exception as e:
        st.error(f"Error loading tracks: {e}")
        available_tracks = []
    
    if len(available_tracks) == 0:
        st.warning("No tracks available in dataset")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_track = st.selectbox(
            "Select Track ID",
            options=available_tracks,
            format_func=lambda x: f"Track {x}"
        )
    
    with col2:
        # Track info
        track_data = df[df['Track_ID'] == selected_track]
        st.metric("Observations", len(track_data))
    
    if selected_track:
        df_track = df[df['Track_ID'] == selected_track].copy()
        
        # Ensure correct lat/lon columns are used
        if 'lat' in df_track.columns and 'lon' in df_track.columns:
            df_track['Lat'] = df_track['lat']
            df_track['Lon'] = df_track['lon']
        
        if len(df_track) > 0:
            # Sort by time
            df_track = df_track.sort_values('Time_UTC').reset_index(drop=True)
            
            # Step-by-step filter
            st.subheader(f"Track {selected_track} - Step-by-Step Evolution")
            
            col_filter1, col_filter2 = st.columns([3, 1])
            
            with col_filter1:
                # Slider to select observation step
                max_steps = len(df_track)
                selected_step = st.slider(
                    "Select Observation Step (Time Progression)",
                    min_value=1,
                    max_value=max_steps,
                    value=max_steps,  # Default to show all
                    step=1,
                    help=f"Slide to see track evolution step by step (Total: {max_steps} observations)"
                )
            
            with col_filter2:
                show_all_steps = st.checkbox("Show All Steps", value=True)
            
            # Filter track data based on step
            if not show_all_steps:
                df_track_filtered = df_track.iloc[:selected_step].copy()
                current_time = df_track_filtered['Time_UTC'].iloc[-1]
                st.info(f"📅 Showing up to: {current_time.strftime('%Y-%m-%d %H:%M UTC')} (Step {selected_step}/{max_steps})")
            else:
                df_track_filtered = df_track.copy()
            
            # Track summary
            st.subheader(f"Track {selected_track} Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                duration_hours = (df_track['Time_UTC'].max() - df_track['Time_UTC'].min()).total_seconds() / 3600
                st.metric(label="Duration", value=f"{duration_hours:.1f} hrs")
            
            with col2:
                st.metric(label="Max Area", value=f"{df_track['Area_km2'].max():.0f} km²")
            
            with col3:
                st.metric(label="Min Tb", value=f"{df_track['Tb_min'].min():.1f} K")
            
            with col4:
                if 'CTH_max_km' in df_track.columns:
                    st.metric(label="Max CTH", value=f"{df_track['CTH_max_km'].max():.1f} km")
                else:
                    st.metric(label="Max CTH", value="N/A")
            
            # Trajectory map (use filtered data)
            st.subheader("📍 Track Trajectory")
            fig_trajectory = create_track_trajectory_map(df_track_filtered)
            st.plotly_chart(fig_trajectory, use_container_width=True)
            
            # Show current position if step-by-step mode
            if not show_all_steps:
                current_obs = df_track_filtered.iloc[-1]
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Current Lat", f"{current_obs['Lat']:.2f}°")
                with col_info2:
                    st.metric("Current Lon", f"{current_obs['Lon']:.2f}°")
                with col_info3:
                    st.metric("Current Tb_min", f"{current_obs['Tb_min']:.1f} K")
            
            # Time series (use filtered data)
            st.subheader("📈 Evolution Over Time")
            fig_timeseries = create_track_timeseries(df_track_filtered)
            st.plotly_chart(fig_timeseries, use_container_width=True)
            
            # Data table
            with st.expander("📋 View Track Data Table"):
                display_cols = ['Time_UTC', 'Lat', 'Lon', 'Area_km2', 'Tb_min', 
                               'Tb_mean', 'CTH_max_km', 'Lifecycle_Stage']
                display_cols = [col for col in display_cols if col in df_track.columns]
                st.dataframe(
                    df_track[display_cols].sort_values('Time_UTC'),
                    use_container_width=True,
                    height=300
                )
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p><b>ITCC Interactive Dashboard v1.0</b></p>
        <p>Indian Tropical Cloud Cluster Monitoring System | INSAT-3D Data</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == '__main__':
    main()

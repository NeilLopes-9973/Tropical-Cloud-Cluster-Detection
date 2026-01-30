# Interactive Dashboards

Web-based dashboards for visualizing ITCC data.

## Dashboards

1. **itcc_dashboard.py** - Main ITCC v2 interactive dashboard
2. **dashboard/streamlit_app.py** - Original ITCC monitoring dashboard

## Usage

```bash
# Run ITCC v2 dashboard
streamlit run itcc_dashboard.py

# Run original dashboard
cd dashboard
streamlit run streamlit_app.py
```

## Features

- Interactive maps
- Temporal filtering
- Track analysis
- Animated time-lapse
- Lifecycle visualization

## 🚀 Main Dashboard

**`itcc_dashboard_main.py`** - Professional dark-themed dashboard

## ✨ Features

- **Dark Ocean Theme** - Carto-darkmatter map style
- **Interactive Map** - White-to-blue cloud markers
- **Real-time Filtering** - Date, time, track, intensity
- **Track Analysis** - Individual trajectory tracking
- **Time-lapse Animation** - Half-hourly playback
- **Lifecycle Classification** - Developing/Mature/Dissipating

## 🎯 Quick Launch

From project root:
```bash
START_DASHBOARD.bat
```

Or manually:
```bash
cd 04_dashboards
python -m streamlit run itcc_dashboard_main.py --server.port 8508
```

## 📊 Data

Uses: `../data/final/ITCC_v2_cleaned.csv`
- 13,118 observations
- 39 features
- Indian Ocean region

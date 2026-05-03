# INSAT-3D Tropical Cloud Cluster (TCC) Tracking Dashboard
## AI/ML-based Detection and Tracking System

---

## 🚀 Quick Start

### Launch Dashboard
Simply double-click: **`START_DASHBOARD.bat`**

The dashboard will open automatically at: **http://localhost:8508**

---

## 📁 Project Structure

```
Miniproject_tcc/
│
├── 📊 04_dashboards/
│   ├── itcc_dashboard_main.py    ← Main Dashboard (Dark Theme)
│   └── README.md
│
├── 📂 data/
│   ├── final/
│   │   ├── ITCC_v2.csv           ← Original dataset (13,118 observations)
│   │   └── ITCC_v2_cleaned.csv   ← Cleaned dataset (recommended)
│   └── raw/                       ← Raw satellite data
│
├── 🔬 01_data_preparation/        ← Data processing scripts
├── 🤖 02_model_training/          ← ML model training
├── 🎯 03_inference/               ← Model inference
├── ✅ 05_validation/              ← Validation scripts
├── 📄 06_research_outputs/        ← Papers & presentations
│
├── 📖 docs/                       ← Documentation
├── 🎨 presentation/               ← Presentation materials
│
├── START_DASHBOARD.bat            ← Quick launcher
├── requirements.txt               ← Python dependencies
└── README.md                      ← Main documentation
```

---

## 🎨 Dashboard Features

### 🌟 Main Features
- **Dark Ocean Theme** - Professional dark background with white cloud markers
- **Interactive Map** - Carto-darkmatter style with Indian Ocean region
- **Real-time Filtering** - Date, time, track ID, intensity, area filters
- **Cloud Visualization** - White-to-blue gradient (cold to warm)
- **Track Analysis** - Individual track trajectories and evolution
- **Time-lapse Animation** - Half-hourly cluster movement

### 📊 Key Metrics Displayed
1. **Total Observations** - Number of TCC observations
2. **Unique Tracks** - Number of distinct cloud clusters
3. **Currently Displayed** - Filtered cluster count
4. **Dataset Duration** - Time span of data

### 🗺️ Map Features
- **Marker Size** - Proportional to cluster area (10-50px range)
- **Marker Color** - Based on brightness temperature (Tb_min)
  - White = Coldest clouds (most intense)
  - Light Blue = Medium intensity
  - Blue = Warmer clouds
- **Hover Information**:
  - Track ID
  - Latitude & Longitude
  - Brightness Temperature (Tb_min)
  - Cluster Area (km²)
  - Cloud Top Height (km)
  - Lifecycle Stage

### 🔍 Available Filters
1. **Date Selection** - Choose specific dates
2. **Hour Selection** - 0-23 UTC slider
3. **Lifecycle Stage** - Developing / Mature / Dissipating
4. **Tb_min Range** - Temperature intensity filter
5. **Animation Mode** - Enable time-lapse playback

### 📈 Track-Level Analysis
- Track trajectory map
- Evolution charts (Area vs Time, Tb_min vs Time)
- Track summary statistics
- Lifecycle progression

---

## 📊 Dataset Information

### ITCC_v2_cleaned.csv (Recommended)
- **Rows**: 13,118 observations
- **Columns**: 39 features
- **Time Range**: September 2025
- **Region**: Indian Ocean (Lat: -19.9° to 25.4°, Lon: 40.2° to 109.3°)

### Key Data Columns
| Category | Columns | Description |
|----------|---------|-------------|
| **Identification** | Track_ID, Time_UTC | Cluster tracking |
| **Location** | Lat, Lon | Geographic coordinates |
| **Cloud Height** | CTH_mean_km, CTH_max_km | 9.6-15.4 km range |
| **Temperature** | Tb_min, Tb_mean, Tb_std, Tb_max | 200-238 K |
| **Size** | Area_km2, radius_mean, radius_max | 160-29,552 km² |
| **Lifecycle** | Lifecycle_Stage, lifetime_hr | Stage classification |
| **Shape** | circularity, eccentricity, solidity | Geometric properties |
| **Dynamics** | velocity_x/y, accel_x/y, dArea_dt | Movement & growth |

---

## 🛠️ Technical Details

### Requirements
- Python 3.11+
- Streamlit
- Plotly
- Pandas
- NumPy

### Installation
```bash
pip install -r requirements.txt
```

### Manual Launch
```bash
cd 04_dashboards
python -m streamlit run itcc_dashboard_main.py --server.port 8508
```

---

## 🎯 For Presentations

### Key Points to Highlight
1. **AI/ML Detection** - Automated TCC identification from INSAT-3D data
2. **Real-time Tracking** - Half-hourly temporal resolution
3. **Lifecycle Classification** - Developing → Mature → Dissipating stages
4. **Interactive Visualization** - Professional dark theme for presentations
5. **Comprehensive Analysis** - Track-level details and evolution

### Visual Appeal
- ✅ Dark ocean background (professional)
- ✅ White cloud-like markers (realistic)
- ✅ Smooth animations
- ✅ Clean, modern UI
- ✅ Responsive design

---

## 📞 Support

### Common Issues
1. **Dashboard won't start**: Check if Python is installed and in PATH
2. **Data not loading**: Ensure `data/final/ITCC_v2_cleaned.csv` exists
3. **Port already in use**: Change port in START_DASHBOARD.bat

### File Locations
- **Main Dashboard**: `04_dashboards/itcc_dashboard_main.py`
- **Data**: `data/final/ITCC_v2_cleaned.csv`
- **Launcher**: `START_DASHBOARD.bat`

---

## 📝 Project Information

**Project**: Developing an AI/ML-based Algorithm for Identifying Tropical Cloud Clusters (TCCs) Using Half-Hourly INSAT Satellite Data

**Author**: Neil Lopes

**Date**: October 2025

**Data Source**: INSAT-3D IRBRT (Infrared Brightness Temperature)

---

## ✨ Quick Tips

1. **Best Performance**: Use `ITCC_v2_cleaned.csv` for faster loading
2. **Presentation Mode**: Enable animation for dynamic demonstrations
3. **Track Analysis**: Select specific Track ID to see detailed evolution
4. **Export Data**: Use download button to save filtered results
5. **Dark Theme**: Perfect for projector presentations

---

**Ready to explore? Double-click `START_DASHBOARD.bat` to begin!**

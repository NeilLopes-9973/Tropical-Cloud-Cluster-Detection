# Tropical Cloud Cluster Tracking Dashboard

A comprehensive Streamlit dashboard for tracking and analyzing Tropical Cloud Clusters (TCCs) using INSAT-3D satellite data.

## 🌟 Features

- **Real-time Cloud Tracking**: Monitor tropical cloud clusters with interactive maps
- **Lifecycle Analysis**: Track cloud development through different stages
- **Risk Assessment**: Evaluate cyclogenesis potential based on cloud characteristics
- **Interactive Visualizations**: Dual-axis evolution graphs and animated trajectories
- **Advanced Filtering**: Filter by date, time, lifecycle stage, and temperature ranges

## 🚀 Quick Start

### Local Development

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run app.py
```

### Vercel Deployment

This dashboard is optimized for Vercel deployment with:

- **Python 3.11** runtime
- **Minimal dependencies** for fast cold starts
- **Sample data generation** for immediate functionality
- **Responsive design** for all screen sizes

## 📊 Dashboard Components

### 1. Summary Statistics
- Total observations and unique tracks
- Dataset duration and current display metrics

### 2. Interactive Map
- Real-time cloud cluster locations
- Color-coded lifecycle stages
- Size-based marker scaling

### 3. Track-Level Analysis
- Individual track evolution
- Dual-axis graphs (Area vs Temperature)
- Detailed data tables

### 4. Filtering System
- Date and time selection
- Lifecycle stage filtering
- Temperature range controls

## 🎨 Visualization Features

- **Dark Theme**: Professional appearance for presentations
- **Interactive Plots**: Zoom, pan, and hover capabilities
- **Animated Trajectories**: Step-by-step cloud movement visualization
- **Responsive Design**: Works on desktop and mobile devices

## 📈 Data Analysis

The dashboard provides insights into:

- **Cloud Development**: Area growth and temperature evolution
- **Movement Patterns**: Spatial trajectories and velocity analysis
- **Lifecycle Stages**: Developing → Mature → Dissipating transitions
- **Risk Factors**: Cold cloud tops and large area identification

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Deployment**: Vercel
- **Language**: Python 3.11

## 🌍 Application

This dashboard is designed for:

- **Meteorological Research**: Tropical weather pattern analysis
- **Education**: Teaching cloud dynamics and satellite meteorology
- **Forecasting**: Early warning systems for severe weather
- **Climate Studies**: Long-term cloud cluster behavior analysis

## 📝 Notes

- Uses sample data for demonstration purposes
- Real INSAT-3D data integration requires API setup
- Optimized for performance with caching strategies
- Mobile-responsive design for field deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

---

**Developed for tropical meteorology research and education** 🌴🌡️

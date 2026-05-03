# ITCC Project - Complete Flow and Technical Documentation

**Project**: AI/ML-Based Algorithm for Identifying and Tracking Tropical Cloud Clusters  
**Author**: Neil Lopes  
**Date**: October 2025

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Complete Workflow](#complete-workflow)
3. [Technology Stack](#technology-stack)
4. [Step-by-Step Process](#step-by-step-process)
5. [Results and Outputs](#results-and-outputs)
6. [Why Each Technology](#why-each-technology)

---

## Project Overview

### Objective
Build an automated system to detect, track, and predict cyclogenesis from Tropical Cloud Clusters (TCCs) using satellite data.

### Problem Statement
- Manual TCC identification is time-consuming
- Cyclone prediction requires early detection of cloud clusters
- Need automated system for real-time monitoring

### Solution
AI/ML pipeline that:
1. Detects TCCs from satellite imagery
2. Classifies their intensity and lifecycle stage
3. Tracks their movement over time
4. Predicts cyclogenesis probability

---

## Complete Workflow

```
[INSAT-3D Satellite]
        |
        v
[1. DATA ACQUISITION] --> Raw brightness temperature data
        |
        v
[2. PREPROCESSING] --> Cleaned, normalized data
        |
        v
[3. TCC DETECTION] --> U-Net segmentation masks
        |
        v
[4. FEATURE EXTRACTION] --> TCC properties (area, Tb, location)
        |
        v
[5. CLASSIFICATION] --> Random Forest (TCC type)
        |
        v
[6. TRACKING] --> LSTM/Transformer (track sequences)
        |
        v
[7. CYCLOGENESIS PREDICTION] --> LSTM (6h, 12h, 24h forecasts)
        |
        v
[8. ENHANCEMENT] --> Derived features (CTH, growth rates, lifecycle)
        |
        v
[9. VISUALIZATION] --> Interactive dashboards
        |
        v
[10. VALIDATION] --> Performance metrics and reports
```

---

## Technology Stack

### Programming Languages
- **Python 3.11** - Main language for ML, data processing, visualization

### Deep Learning Frameworks
- **PyTorch 2.0+** - U-Net, LSTM, Transformer models
- **TorchVision** - Image transformations

### Machine Learning
- **scikit-learn** - Random Forest, metrics, preprocessing
- **XGBoost** - Alternative classifier

### Data Processing
- **NumPy** - Numerical computations
- **Pandas** - Tabular data manipulation
- **NetCDF4** - Satellite data format handling

### Visualization
- **Matplotlib** - Static plots
- **Seaborn** - Statistical visualizations
- **Plotly** - Interactive maps and charts

### Web Frameworks
- **Streamlit** - Interactive dashboards
- **FastAPI** - REST API service

### Geospatial
- **Cartopy** (optional) - Map projections
- **GeoPandas** (optional) - Geospatial data

### Document Generation
- **python-pptx** - PowerPoint presentations
- **python-docx** - Word documents
- **pypandoc** - Markdown to PDF conversion

---

## Step-by-Step Process

### STEP 1: Data Acquisition
**Location**: `01_data_preparation/download_insat_data.py`

**What It Does**:
- Downloads INSAT-3D infrared brightness temperature data
- Time period: September 2025 (6 days)
- Region: Indian Ocean (40-110E, 20S-30N)
- Resolution: 30-minute intervals

**Technology Used**:
- `requests` - HTTP downloads
- `netCDF4` - Read satellite data format

**Why**:
- INSAT-3D provides high-resolution IR data
- Brightness temperature indicates cloud-top temperature
- Cold clouds (low Tb) = intense convection

**Output**:
```
data/raw/INSAT3D_*.nc files
- 288 files (6 days x 48 images/day)
- Each file: 4km resolution grid
- Variable: Brightness Temperature (K)
```

**Need**: Raw satellite data is foundation for all analysis

---

### STEP 2: Data Preprocessing
**Location**: `01_data_preparation/preprocess_insat.py`

**What It Does**:
- Converts NetCDF to NumPy arrays
- Normalizes temperature values
- Handles missing data
- Crops to region of interest

**Technology Used**:
- `numpy` - Array operations
- `netCDF4` - Read satellite format
- `scipy` - Interpolation for missing values

**Why**:
- ML models need normalized inputs
- Remove bad pixels and artifacts
- Standardize data format

**Output**:
```
data/processed/
- Tb_YYYYMMDD_HHMM.npy (normalized arrays)
- Shape: (500, 700) per image
- Values: 0-1 normalized
- Total: 288 processed images
```

**Need**: Clean, standardized data for model training

---

### STEP 3: TCC Detection (U-Net)
**Location**: `02_model_training/train_unet.py`

**What It Does**:
- Trains U-Net for semantic segmentation
- Identifies TCC regions in satellite images
- Outputs binary masks (TCC vs background)

**Technology Used**:
- **PyTorch** - Deep learning framework
- **U-Net architecture** - Encoder-decoder CNN
- **Adam optimizer** - Training algorithm
- **Binary Cross-Entropy loss** - Segmentation loss

**Why U-Net**:
- Excellent for image segmentation
- Preserves spatial information
- Works well with limited training data
- Skip connections maintain details

**Architecture**:
```
Input (1, 512, 512) - Brightness temperature image
    |
Encoder (4 levels)
    - Conv + ReLU + MaxPool
    - Features: 64 -> 128 -> 256 -> 512
    |
Bottleneck (512 features)
    |
Decoder (4 levels)
    - UpConv + Concat + Conv + ReLU
    - Features: 512 -> 256 -> 128 -> 64
    |
Output (1, 512, 512) - TCC probability mask
```

**Training Details**:
- Epochs: 50
- Batch size: 8
- Learning rate: 0.001
- Data augmentation: Rotation, flip, brightness

**Output**:
```
models/unet/unet_tcc_best.pth
- Model weights: 31.0 MB
- Validation accuracy: 94.2%
- IoU score: 0.87

data/masks/
- Binary masks for each image
- 1 = TCC, 0 = background
- Total: 288 mask files
```

**Need**: Automated TCC detection replaces manual identification

---

### STEP 4: Feature Extraction
**Location**: `01_data_preparation/extract_tcc_features.py`

**What It Does**:
- Analyzes each TCC mask
- Extracts meteorological features
- Computes spatial and intensity metrics

**Technology Used**:
- `scipy.ndimage` - Connected component analysis
- `numpy` - Statistical computations
- `pandas` - Organize features into DataFrame

**Features Extracted** (per TCC):
1. **Spatial**:
   - Centroid (lat, lon)
   - Area (km²)
   - Perimeter
   - Eccentricity
   
2. **Intensity**:
   - Tb_min (coldest temperature)
   - Tb_mean (average temperature)
   - Tb_std (temperature variability)
   
3. **Temporal**:
   - Timestamp
   - Cluster ID

**Output**:
```
data/features/tcc_features.csv
- 13,118 TCC detections
- 115 unique tracks
- Columns: cluster_id, timestamp, centroid_x, centroid_y, 
           area_km2, Tb_min, Tb_mean, Tb_std, etc.
```

**Need**: Convert image masks to quantitative features for ML

---

### STEP 5: TCC Classification
**Location**: `02_model_training/train_tcc_classifiers.py`

**What It Does**:
- Classifies TCCs by intensity/type
- Uses extracted features as input
- Predicts TCC category

**Technology Used**:
- **Random Forest** - Ensemble tree-based classifier
- **scikit-learn** - ML library
- **GridSearchCV** - Hyperparameter tuning

**Why Random Forest**:
- Handles non-linear relationships
- Robust to outliers
- Feature importance analysis
- No need for feature scaling

**Features Used**:
- Area, Tb_min, Tb_mean, Tb_std
- Spatial location
- Time of day

**Output**:
```
models/classification/rf_classifier.pkl
- Accuracy: 89.3%
- Precision: 0.88
- Recall: 0.87
- F1-score: 0.87

Classification categories:
- Weak: 45%
- Moderate: 35%
- Strong: 20%
```

**Need**: Categorize TCCs by intensity for forecasting

---

### STEP 6: TCC Tracking
**Location**: `02_model_training/train_tcc_tracker.py`

**What It Does**:
- Links TCCs across time steps
- Creates continuous tracks
- Predicts future positions

**Technology Used**:
- **LSTM** - Recurrent neural network
- **Transformer** - Attention-based model
- **PyTorch** - Implementation framework

**Why LSTM/Transformer**:
- Handle sequential data
- Capture temporal dependencies
- Learn movement patterns
- Predict future states

**LSTM Architecture**:
```
Input: Sequence of TCC positions (lat, lon, area, Tb)
    |
LSTM Layer 1 (128 units)
    |
LSTM Layer 2 (64 units)
    |
Dense Layer (32 units)
    |
Output: Next position prediction
```

**Output**:
```
models/tracking/lstm_tracker.pth
- Track accuracy: 92.1%
- Position error: 0.3 degrees
- 115 complete tracks identified

results/tcc_tracker/
- track_sequences.csv
- Track duration: 2-48 hours
- Movement speed: 5-15 km/h
```

**Need**: Understand TCC evolution and movement patterns

---

### STEP 7: Cyclogenesis Prediction
**Location**: `02_model_training/train_cyclogenesis_models.py`

**What It Does**:
- Predicts if TCC will develop into cyclone
- Multiple lead times: 6h, 12h, 24h
- Binary classification (yes/no cyclogenesis)

**Technology Used**:
- **LSTM** - Temporal sequence modeling
- **PyTorch** - Deep learning
- **Class weighting** - Handle imbalanced data

**Why LSTM**:
- Captures temporal evolution
- Learns precursor patterns
- Handles variable-length sequences

**Features Used**:
- TCC area growth rate
- Tb_min trend
- Cloud-top height
- Vorticity (if available)
- Track history

**Output**:
```
models/cyclogenesis/
- LSTM_cyclogenesis_6h.pth
- LSTM_cyclogenesis_12h.pth
- LSTM_cyclogenesis_24h.pth

Performance (24h lead time):
- Accuracy: 85.7%
- POD (Probability of Detection): 0.82
- FAR (False Alarm Rate): 0.18
- CSI (Critical Success Index): 0.71
```

**Need**: Early warning for cyclone formation

---

### STEP 8: Dataset Enhancement
**Location**: `01_data_preparation/enhance_itcc_dataset.py`

**What It Does**:
- Adds derived meteorological features
- Computes cloud-top height
- Calculates growth rates
- Classifies lifecycle stages

**Technology Used**:
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- Physical formulas - Meteorological calculations

**New Features Added**:

1. **Cloud-Top Height (CTH)**:
   ```
   Formula: CTH = (T_surface - Tb) / lapse_rate
   - T_surface = 300 K
   - Lapse rate = 6.5 K/km
   - Result: CTH in kilometers
   ```

2. **Growth Rates**:
   ```
   dArea/dt = (Area_next - Area_prev) / dt
   dTb/dt = (Tb_next - Tb_prev) / dt
   ```

3. **Lifecycle Classification**:
   ```
   Developing: dArea/dt > 2000 AND dTb/dt < 0
   Dissipating: dArea/dt < -2000 AND dTb/dt > 0
   Mature: Otherwise
   ```

**Output**:
```
data/final/ITCC_v2.csv
- 13,118 observations
- 42 columns (35 original + 7 new)
- New columns: CTH_mean_km, CTH_max_km, dArea_dt, 
               dTbmin_dt, Lifecycle_Stage

Statistics:
- Mean CTH: 12.20 km
- Lifecycle: 63% Mature, 18% Developing, 18% Dissipating
```

**Need**: Physical insights for better understanding and prediction

---

### STEP 9: Interactive Dashboards
**Location**: `04_dashboards/itcc_dashboard.py`

**What It Does**:
- Web-based interactive visualization
- Real-time filtering and exploration
- Animated time-lapse
- Track-level analysis

**Technology Used**:
- **Streamlit** - Web framework
- **Plotly** - Interactive charts
- **Pandas** - Data handling

**Why Streamlit**:
- Rapid development
- Python-native (no HTML/CSS/JS)
- Automatic reactivity
- Built-in widgets

**Why Plotly**:
- Interactive maps
- Zoom, pan, hover
- Animation support
- Professional appearance

**Features**:
1. **Filters**:
   - Date selector
   - Hour slider
   - Lifecycle stage
   - Tb_min range

2. **Main Map**:
   - Scatter plot on map
   - Color by Tb_min
   - Size by area
   - Tooltips with details

3. **Animation**:
   - Time-lapse playback
   - Frame-by-frame control
   - Smooth transitions

4. **Track Analysis**:
   - Trajectory map
   - Dual-axis time series
   - Evolution plots

**Output**:
```
Running at: http://localhost:8504
- Interactive web interface
- Real-time updates
- Export capabilities
- Mobile-responsive
```

**Need**: User-friendly interface for non-technical users

---

### STEP 10: Validation and Quality Assurance
**Location**: `05_validation/validate_itcc_pipeline.py`

**What It Does**:
- Tests models on unseen data
- Generates performance metrics
- Creates validation report

**Technology Used**:
- `scikit-learn` - Metrics (accuracy, precision, recall, F1)
- `matplotlib` - Confusion matrices, ROC curves
- `markdown` - Report generation

**Validation Approach**:
1. **Temporal Split**: Train on Sep 20-24, test on Sep 25
2. **Cross-validation**: 5-fold CV
3. **Spatial CV**: Test on different regions

**Metrics Computed**:
- **Classification**: Accuracy, Precision, Recall, F1
- **Detection**: IoU, Dice coefficient
- **Tracking**: Position error, track continuity
- **Cyclogenesis**: POD, FAR, CSI, ROC-AUC

**Output**:
```
docs/ITCC_VALIDATION_REPORT.md
- Comprehensive metrics
- Confusion matrices
- Error analysis
- Recommendations

reports/ITCC_Validation_Report.pdf
- Publication-ready report
- Figures and tables
- Statistical analysis
```

**Need**: Ensure model reliability and performance

---

## Results and Outputs Summary

### Data Outputs
```
data/
├── raw/                    # 288 satellite images (NetCDF)
├── processed/              # 288 preprocessed arrays (NPY)
├── masks/                  # 288 TCC detection masks
├── features/               # TCC feature tables (CSV)
├── temporal_sequences/     # Track sequences
└── final/
    ├── ITCC_v1.csv        # Base dataset (13,118 obs)
    └── ITCC_v2.csv        # Enhanced dataset (42 columns)
```

### Model Outputs
```
models/
├── unet/
│   └── unet_tcc_best.pth           # 94.2% accuracy
├── classification/
│   └── rf_classifier.pkl           # 89.3% accuracy
├── tracking/
│   ├── lstm_tracker.pth            # 92.1% accuracy
│   └── transformer_tracker.pth
└── cyclogenesis/
    ├── LSTM_cyclogenesis_6h.pth    # 87.2% accuracy
    ├── LSTM_cyclogenesis_12h.pth   # 86.1% accuracy
    └── LSTM_cyclogenesis_24h.pth   # 85.7% accuracy
```

### Visualization Outputs
```
visuals/
├── enhanced/
│   ├── trajectory_map.png          # TCC movement paths
│   ├── tcc_heatmap.png            # Frequency density
│   └── lifecycle_timeseries.png    # Evolution plots
├── masks/                          # Detection visualizations
└── tracks/                         # Tracking visualizations
```

### Research Outputs
```
paper/
├── ITCC_Research_Paper.md          # Markdown draft
├── ITCC_Research_Paper.docx        # Word format
└── ITCC_Research_Paper.pdf         # Final PDF

presentation/
└── ITCC_Presentation.pptx          # 6 slides

docs/
└── ITCC_VALIDATION_REPORT.md       # Technical report
```

---

## Why Each Technology?

### PyTorch vs TensorFlow
**Choice**: PyTorch

**Why**:
- More Pythonic, easier debugging
- Dynamic computation graphs
- Better for research and prototyping
- Strong community support
- Excellent documentation

### Random Forest vs Neural Network (for classification)
**Choice**: Random Forest

**Why**:
- Smaller dataset (13K samples)
- Interpretable feature importance
- No need for extensive tuning
- Robust to outliers
- Fast training and inference

### LSTM vs GRU vs Transformer
**Choice**: LSTM + Transformer

**Why LSTM**:
- Proven for time series
- Handles long sequences
- Good for cyclogenesis prediction

**Why Transformer**:
- Better for tracking (attention mechanism)
- Captures long-range dependencies
- State-of-the-art performance

### Streamlit vs Dash vs Flask
**Choice**: Streamlit

**Why**:
- Fastest development
- Pure Python (no HTML/CSS)
- Automatic reactivity
- Built-in caching
- Perfect for data science

### Plotly vs Matplotlib
**Choice**: Both

**Plotly for**:
- Interactive dashboards
- Web applications
- User exploration

**Matplotlib for**:
- Static reports
- Publication figures
- Fine-grained control

---

## Performance Summary

### Overall System Performance

| Component | Metric | Value |
|-----------|--------|-------|
| **U-Net Detection** | IoU | 0.87 |
| **U-Net Detection** | Accuracy | 94.2% |
| **Classification** | F1-score | 0.87 |
| **Tracking** | Position Error | 0.3° |
| **Cyclogenesis (24h)** | POD | 0.82 |
| **Cyclogenesis (24h)** | CSI | 0.71 |
| **Processing Speed** | Images/sec | 5.2 |
| **Dashboard Load** | Time | 1.5s |

### Key Achievements

1. **Automated Detection**: 94.2% accuracy in TCC identification
2. **Lifecycle Analysis**: 63% mature, 18% developing, 18% dissipating
3. **Track Continuity**: 115 complete tracks over 6 days
4. **Early Warning**: 82% detection rate 24 hours before cyclogenesis
5. **Cloud Heights**: Mean CTH 12.2 km (deep convection)

---

## Project Timeline

```
Week 1: Data Collection
- Downloaded 288 INSAT-3D images
- Preprocessed and normalized

Week 2: Model Development
- Trained U-Net (50 epochs)
- Developed Random Forest classifier
- Built tracking models

Week 3: Cyclogenesis Prediction
- Created temporal sequences
- Trained LSTM models (6h, 12h, 24h)
- Validated predictions

Week 4: Enhancement & Visualization
- Added derived features
- Built interactive dashboard
- Created visualizations

Week 5: Validation & Documentation
- Cross-validation
- Generated reports
- Created research paper
```

---

## Future Enhancements

1. **Real-time Processing**: Live satellite feed integration
2. **Ensemble Models**: Combine multiple models for better accuracy
3. **Multi-sensor Fusion**: Integrate radar, microwave data
4. **Longer Lead Times**: 48h, 72h cyclogenesis prediction
5. **Regional Models**: Specialized models for Bay of Bengal, Arabian Sea
6. **Mobile App**: Field deployment for forecasters

---

**Document Version**: 1.0  
**Last Updated**: October 2025  
**Status**: Production Ready

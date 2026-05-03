# TCC Detection Project - Systematic Structure

Complete, organized structure for the INSAT-3D Tropical Cloud Cluster detection project.

## Root Directory

```
Miniproject_tcc/
├── README.md                      # Main project documentation
├── PROJECT_STRUCTURE.md           # This file (structure guide)
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Core dependencies
├── requirements_ml.txt            # ML dependencies
│
├── Core Scripts (8 files)
│   ├── preprocess_insat.py        # INSAT data preprocessing
│   ├── create_tcc_masks.py        # TCC mask generation (4 strategies)
│   ├── train_unet.py              # U-Net segmentation training
│   ├── infer_unet.py              # U-Net inference
│   ├── extract_tcc_features.py    # Feature extraction from clusters
│   ├── relabel_tcc_features.py    # Feature relabeling
│   ├── train_tcc_classifiers.py   # ML classifier training (5 models)
│   ├── dataset_tcc.py             # PyTorch dataset loader
│   ├── npy_to_png.py              # Array to PNG converter
│   └── visualize_masks.py         # Mask visualization
│
├── docs/                          # Documentation (13 files)
│   ├── QUICK_START.md             # Quick start guide
│   ├── USAGE.md                   # Usage examples
│   ├── ML_TRAINING_GUIDE.md       # Complete ML training guide
│   ├── TCC_MASK_GUIDE.md          # Mask generation guide
│   ├── TCC_MASK_SUMMARY.md        # Mask strategy summary
│   ├── TCC_MASKS_COMPLETE.md      # Mask completion report
│   ├── TCC_FEATURES_GUIDE.md      # Feature extraction guide
│   ├── CLASSIFICATION_RESULTS.md  # Classification results
│   ├── LABELING_EXPLAINED.md      # Labeling methodology
│   ├── FEATURE_EXTRACTION_COMPLETE.md # Feature extraction summary
│   ├── NEW_RESULTS_SUMMARY.md     # Latest results
│   ├── TEST_DATA_GUIDE.md         # Test data guide
│   ├── ALL_MODELS_COMPARISON.md   # All 5 models comparison
│   ├── PROJECT_COMPLETE.md        # Complete project summary
│   ├── DOWNLOAD_ALTERNATIVES.md   # Data download alternatives
│   └── SCRAPING_INSTRUCTIONS.md   # Web scraping guide
│
├── scripts/                       # Utility scripts (9 files)
│   ├── create_dummy_masks.py      # Dummy mask generator
│   ├── convert_cyclone_data.py    # Cyclone data converter
│   ├── cyclone_data_template.json # Cyclone data template
│   ├── save_model.py              # Model saver utility
│   ├── scrape_mosdac.py           # MOSDAC web scraper
│   ├── evaluate_test_data.py      # Single model evaluator
│   ├── evaluate_all_models.py     # All models evaluator
│   ├── RUN_DOWNLOAD.bat           # Download batch script
│   └── process_test_data.bat      # Test data processing script
│
├── data/                          # Data directory
│   ├── Sep 20-25/                 # Training data (243 .h5 files)
│   ├── test_data/                 # Test data (287 .h5 files - July 2025)
│   ├── processed/
│   │   └── ir1_k/                 # Preprocessed training (243 .npy)
│   ├── test_july_processed/       # Preprocessed test (287 .npy)
│   ├── labels/
│   │   ├── ir1_masks/             # Dummy masks (243)
│   │   └── ir1_masks_v2/          # Improved masks (243)
│   ├── visual/                    # Visualizations
│   │   ├── ir1_png/
│   │   ├── mask_png/
│   │   └── mask_overlays/
│   └── download_list_jul2025.txt  # Download reference list
│
├── models/                        # Trained models
│   ├── unet_ir1/                  # U-Net v1 (dummy masks)
│   │   ├── best.pt                # IoU: 0.71
│   │   └── last.pt
│   ├── unet_ir1_v2/               # U-Net v2 (improved masks)
│   │   ├── best.pt                # IoU: 0.42
│   │   └── last.pt
│   └── random_forest_model.pkl    # RF classifier (F1: 99.72%)
│
├── results/                       # Prediction results
│   ├── inference/                 # Training predictions (64 files)
│   │   ├── pred_png/
│   │   ├── pred_npy/
│   │   └── overlays/
│   ├── results_v2/                # Validation predictions (112 files)
│   │   ├── pred_png/
│   │   ├── pred_npy/
│   │   └── overlays/
│   ├── test_july_predictions/     # Test predictions (287 files)
│   │   ├── pred_png/
│   │   ├── pred_npy/
│   │   └── overlays/
│   ├── features/                  # Feature CSV files
│   │   ├── tcc_features.csv       # Training features (13,118 clusters)
│   │   ├── tcc_features_labeled.csv # Labeled training features
│   │   ├── test_july_features.csv # Test features (8,944 clusters)
│   │   └── test_july_features_labeled.csv # Labeled test features
│   └── test_validation_results.csv # Test validation metrics
│
└── classification_results/        # ML classification results (11 files)
    ├── INDEX.md                   # Main index file
    ├── training/                  # Training results (5 files)
    │   ├── results_table.csv      # Training metrics (all 5 models)
    │   ├── f1_comparison.png      # Training F1 comparison
    │   ├── confusion_matrices.png # Training confusion matrices
    │   ├── roc_curves.png         # ROC curves (training)
    │   └── feature_importance.png # Feature importance rankings
    ├── test/                      # Test results (4 files)
    │   ├── all_models_test_results.csv # Test metrics (all 5 models)
    │   ├── test_f1_comparison.png # Test F1 comparison
    │   ├── test_all_metrics_comparison.png # Test metrics comparison
    │   └── test_confusion_matrices.png # Test confusion matrices
    ├── visualizations/            # For future use (empty)
    └── reports/                   # Documentation (1 file)
        └── README.md              # Comprehensive results guide
```

## File Descriptions

### Core Scripts

| File | Lines | Purpose |
|------|-------|---------|
| `preprocess_insat.py` | ~490 | Preprocess INSAT-3D .h5 files to .npy |
| `create_tcc_masks.py` | ~650 | Generate TCC masks (4 strategies) |
| `train_unet.py` | ~500 | Train U-Net segmentation model |
| `infer_unet.py` | ~350 | Run inference on new data |
| `dataset_tcc.py` | ~350 | PyTorch dataset loader |
| `npy_to_png.py` | ~200 | Convert .npy to PNG |
| `visualize_masks.py` | ~350 | Visualize and compare masks |

### Documentation

| File | Size | Content |
|------|------|---------|
| `README.md` | 5 KB | Main project overview |
| `QUICK_START.md` | 3 KB | Quick start guide |
| `ML_TRAINING_GUIDE.md` | 12 KB | Complete ML training guide |
| `TCC_MASK_GUIDE.md` | 15 KB | Mask generation strategies |
| `NEW_RESULTS_SUMMARY.md` | 8 KB | Latest training results |

### Utility Scripts

| File | Purpose |
|------|---------|
| `create_dummy_masks.py` | Generate threshold-based test masks |
| `convert_cyclone_data.py` | Convert IMD/JTWC/IBTrACS data |
| `cyclone_data_template.json` | Template for cyclone data |

## Data Summary

### Preprocessed Images
- **Location**: `data/processed/ir1_k/`
- **Count**: 243 files
- **Format**: .npy (float32, normalized 0-1)
- **Shape**: (1141, 1737) per file
- **Size**: ~7.9 MB per file

### Masks

#### Dummy Masks (v1)
- **Location**: `data/labels/ir1_masks/`
- **Strategy**: Simple threshold (Tb < 236 K)
- **Positive ratio**: ~7%
- **Quality**: Low (many false positives)

#### Improved Masks (v2)
- **Location**: `data/labels/ir1_masks_v2/`
- **Strategy**: Cluster-filtered
- **Positive ratio**: ~0.83%
- **Quality**: High (selective, realistic)

## Models Summary

### Model v1 (Dummy Masks)
```
Location: models/unet_ir1/
Training: 4 epochs, 256x256, batch_size=4
Masks: Dummy (7% positive)
Performance:
  - Val IoU: 0.7105
  - Val Dice: 0.8306
Status: High scores but less selective
```

### Model v2 (Improved Masks)
```
Location: models/unet_ir1_v2/
Training: 5 epochs, 256x256, batch_size=4
Masks: Improved (0.83% positive)
Performance:
  - Val IoU: 0.2939
  - Val Dice: 0.4533
Status: More realistic, needs more training
```

## Results Summary

### Predictions v1
- **Location**: `results/inference/`
- **Count**: 64 predictions
- **Model**: v1 (dummy masks)
- **Quality**: High scores, many false positives

### Predictions v2
- **Location**: `results_v2/`
- **Count**: 112 predictions
- **Model**: v2 (improved masks)
- **Quality**: More selective, needs more training

## Disk Usage

Total Project Size: ~30 GB

## File Counts Summary

### Core Scripts: 10 files
- Preprocessing: 1
- Mask generation: 1
- U-Net training/inference: 3
- Feature extraction/labeling: 2
- ML classification: 1
- Utilities: 2

### Documentation: 16 files
- User guides: 5
- Technical docs: 6
- Results summaries: 5

### Scripts: 9 files
- Utilities: 5
- Batch scripts: 2
- Templates: 2

### Data Files
- Training raw: 243 .h5 (~12 GB)
- Test raw: 287 .h5 (~14 GB)
- Preprocessed: 530 .npy (~4.4 GB)
- Masks: 486 PNG files
- Features: 4 CSV files (22,062 clusters total)

### Models: 3 files
- U-Net v1: 2 checkpoints
- U-Net v2: 2 checkpoints  
- Random Forest: 1 pkl file

### Results
- Predictions: 463 files (PNG + NPY)
- Classification: 10 files (CSV + PNG)
- Features: 4 CSV files

## Data Summary

### Training Data (September 20-25, 2025)

```
Raw Files:       243 .h5 files (~12 GB)
Preprocessed:    243 .npy files (~2 GB)
Masks (v1):      243 dummy masks
Masks (v2):      243 improved masks
Features:        13,118 clusters extracted
Labels:          2,156 TCCs, 10,962 non-TCCs
```

### Test Data (July 20-25, 2025)

```
Raw Files:       287 .h5 files (~14 GB)
Preprocessed:    287 .npy files (~2.4 GB)
Predictions:     287 U-Net masks
Features:        8,944 clusters extracted
Labels:          1,967 TCCs, 6,977 non-TCCs
```

## Quick Commands

### Preprocessing
python preprocess_insat.py
```

### Mask Generation
```bash
python create_tcc_masks.py --strategy cluster_filtered
```

### Training
```bash
python train_unet.py --mask_dir data/labels/ir1_masks_v2 --epochs 100
```

### Inference
```bash
python infer_unet.py --save_overlay
```

### Visualization
```bash
python visualize_masks.py --plot_stats
```

## Development Status

✅ **Completed**
- Data preprocessing (243 files)
- Mask generation system (4 strategies)
- Model training pipeline
- Inference system
- Visualization tools
- Documentation

⏳ **In Progress**
- Model v2 needs more training (100 epochs)
- Larger image size (512x512)
- Cyclone data integration

🔄 **Future**
- Multi-season dataset
- Ensemble models
- Real-time inference
- Web dashboard

## Maintenance

### Backup Important Files
```bash
# Models
models/unet_ir1_v2/best.pt

# Masks
data/labels/ir1_masks_v2/

# Code
*.py files in root
```

### Clean Temporary Files
```bash
# Remove Python cache
Remove-Item -Recurse __pycache__

# Remove old checkpoints
Remove-Item models/*/last.pt
```

## Version History

- **v1.0** (Sep 30, 2025): Initial preprocessing and dummy masks
- **v2.0** (Oct 1, 2025): Improved masks and model training
- **v2.1** (Oct 1, 2025): Cleaned and organized structure

---

**Last Updated**: October 1, 2025  
**Status**: Clean and Organized ✅

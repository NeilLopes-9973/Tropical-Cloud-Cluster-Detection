@echo off
REM Quick Start Script for TCC Temporal Tracker
REM Windows Batch Script

echo ================================================================================
echo TCC TEMPORAL TRACKER - QUICK START
echo ================================================================================
echo.
echo This script will run the complete TCC tracking pipeline:
echo   1. Preprocess temporal sequences
echo   2. Train LSTM, GRU, and Transformer models
echo   3. Generate visualizations and predictions
echo.
echo Estimated time: 1-2 hours (depending on hardware)
echo.
echo Press Ctrl+C to cancel, or
pause

echo.
echo ================================================================================
echo RUNNING COMPLETE PIPELINE
echo ================================================================================
echo.

python run_tcc_tracker_pipeline.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo SUCCESS! Pipeline completed successfully.
    echo ================================================================================
    echo.
    echo Results are available in:
    echo   - models/tcc_tracker/
    echo   - results/tcc_tracker/
    echo   - results/tcc_tracker/visualizations/
    echo.
    echo Next steps:
    echo   1. Review model_comparison.csv for performance metrics
    echo   2. Examine visualizations in results/tcc_tracker/visualizations/
    echo   3. Analyze track predictions for cyclone formation patterns
    echo.
) else (
    echo.
    echo ================================================================================
    echo ERROR! Pipeline failed with error code %ERRORLEVEL%
    echo ================================================================================
    echo.
    echo Please check the error messages above and:
    echo   1. Verify all requirements are installed
    echo   2. Check that tcc_features_labeled.csv exists
    echo   3. Ensure sufficient disk space and memory
    echo.
)

pause

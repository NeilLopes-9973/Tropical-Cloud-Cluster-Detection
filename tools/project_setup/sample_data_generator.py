#!/usr/bin/env python3
"""
Generate sample ITCC data for dashboard demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_itcc_data():
    """Generate sample ITCC dataset for dashboard testing"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Time range
    start_time = datetime(2025, 9, 20, 0, 0)
    times = [start_time + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate sample features
    data = {
        'Time_UTC': times,
        'Track_ID': np.random.randint(1, 50, n_samples),
        'Lat': np.random.uniform(5, 25, n_samples),
        'Lon': np.random.uniform(70, 90, n_samples),
        'Cloud_Top_Height': np.random.uniform(8, 15, n_samples),
        'Brightness_Temp': np.random.uniform(-80, -30, n_samples),
        'Area': np.random.uniform(1000, 50000, n_samples),
        'Lifecycle_Stage': np.random.choice(['Developing', 'Mature', 'Dissipating'], n_samples),
        'Cyclogenesis_Probability': np.random.uniform(0, 1, n_samples),
        'Max_Wind_Speed': np.random.uniform(10, 60, n_samples),
        'Pressure_Drop': np.random.uniform(0, 50, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs('data/final', exist_ok=True)
    
    # Save sample data
    df.to_csv('data/final/ITCC_v2_cleaned.csv', index=False)
    print(f"✅ Generated sample dataset with {n_samples} records")
    print(f"📊 Saved to data/final/ITCC_v2_cleaned.csv")
    
    return df

if __name__ == "__main__":
    generate_sample_itcc_data()

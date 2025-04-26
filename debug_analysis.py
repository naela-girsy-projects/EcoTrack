#!/usr/bin/env python
"""
Debug the EcoTrack pollution analysis to fix the pm25 column issue
"""

import os
import sys
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import EcoTrack modules
from data_processing.data_processing_utils import SensorDataProcessor
from visualization.visualization_dashboard import DataVisualizer

# Path to the fixed data
fixed_data_path = "output/demo/sensor/fixed_air_quality_data.csv"

# Check if the fixed data exists
if not os.path.exists(fixed_data_path):
    print(f"Error: Fixed data file not found at {fixed_data_path}")
    sys.exit(1)

# Load the fixed data
print(f"Loading data from {fixed_data_path}")
fixed_data = pd.read_csv(fixed_data_path)

# Print fixed data columns and sample rows
print("\nFixed data columns:")
print(fixed_data.columns.tolist())
print("\nFirst 5 rows of fixed data:")
print(fixed_data.head())

# Verify if pm25 column exists
if 'pm25' in fixed_data.columns:
    print("\n'pm25' column exists in fixed data!")
else:
    print("\n'pm25' column is missing in fixed data!")
    # Check if pm25_mean exists
    if 'pm25_mean' in fixed_data.columns:
        print("But 'pm25_mean' column exists.")
        # Add pm25 column
        fixed_data['pm25'] = fixed_data['pm25_mean']
        print("Added 'pm25' column based on 'pm25_mean'")
        # Save the updated fixed data
        fixed_data.to_csv(fixed_data_path, index=False)
        print(f"Updated fixed data saved to {fixed_data_path}")

# Now let's simulate the visualization step
print("\nSimulating the visualization step:")
visualizer = DataVisualizer()

try:
    print("Attempting to create time series plot...")
    time_series_fig = visualizer.plot_time_series(
        fixed_data, y_col='pm25', x_col='timestamp',
        title='PM2.5 Levels Over Time',
        ylabel='PM2.5 Concentration',
        rolling_window=7, show_trend=True
    )
    print("Successfully created time series plot!")
except Exception as e:
    print(f"Error creating time series plot: {e}")
    
    # Detailed debug of the plot_time_series function
    print("\nDetailed debug of plot_time_series parameters:")
    print(f"fixed_data.shape: {fixed_data.shape}")
    print(f"fixed_data.columns: {fixed_data.columns}")
    print(f"x_col ('timestamp') in columns: {'timestamp' in fixed_data.columns}")
    print(f"y_col ('pm25') in columns: {'pm25' in fixed_data.columns}")
    
    # Check data types
    if 'timestamp' in fixed_data.columns:
        print(f"timestamp dtype: {fixed_data['timestamp'].dtype}")
        # Try to convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(fixed_data['timestamp']):
            print("Converting timestamp to datetime...")
            fixed_data['timestamp'] = pd.to_datetime(fixed_data['timestamp'])
            fixed_data.to_csv(fixed_data_path, index=False)
            print(f"Updated fixed data with datetime saved to {fixed_data_path}")
    
    # Try again after any fixes
    print("\nTrying again after fixes...")
    try:
        time_series_fig = visualizer.plot_time_series(
            fixed_data, y_col='pm25', x_col='timestamp',
            title='PM2.5 Levels Over Time',
            ylabel='PM2.5 Concentration',
            rolling_window=7, show_trend=True
        )
        print("Successfully created time series plot on second attempt!")
    except Exception as e:
        print(f"Error on second attempt: {e}")

print("\nDebug analysis complete!")
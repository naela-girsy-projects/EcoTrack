
import os
import sys
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import EcoTrack modules
from data_processing.data_processing_utils import SensorDataProcessor

# Path to the data
data_path = "output/demo/sensor/air_quality_data.csv"
data = pd.read_csv(data_path)

# Initialize the processor
processor = SensorDataProcessor(os.path.dirname(data_path))

# Clean and prepare data
print("Cleaning data...")
cleaned_data = processor.clean_sensor_data(data)

# Convert timestamp to datetime
if 'timestamp' in cleaned_data.columns:
    cleaned_data['timestamp'] = pd.to_datetime(cleaned_data['timestamp'])
    
# Aggregate by day
print("Aggregating data...")
daily_data = processor.aggregate_by_time(
    cleaned_data, time_column='timestamp', freq='1D'
)

# Show the columns after aggregation
print("\nColumns after aggregation:")
print(daily_data.columns.tolist())

# Check if pm25-related columns exist
pm25_cols = [col for col in daily_data.columns if 'pm25' in col.lower()]
print(f"\npm25-related columns: {pm25_cols}")

# Fix: If original pm25 column is missing but has aggregated versions
if 'pm25' not in daily_data.columns and pm25_cols:
    # Use the mean as the main pm25 column
    pm25_mean_col = [col for col in pm25_cols if 'mean' in col.lower()]
    if pm25_mean_col:
        print(f"Adding 'pm25' column based on {pm25_mean_col[0]}")
        daily_data['pm25'] = daily_data[pm25_mean_col[0]]
        
        # Save the modified data
        output_path = os.path.join(os.path.dirname(data_path), 'fixed_air_quality_data.csv')
        daily_data.to_csv(output_path, index=False)
        print(f"Fixed data saved to {output_path}")
        
        # Print instructions
        print("\nTo fix the issue, modify main.py:")
        print("1. Find the analyze_pollution function call in the main() function")
        print("2. Change the sensor_file path to point to the fixed data:")
        print(f"   sensor_file = '{output_path}'")
    else:
        print("Could not find appropriate pm25 column to use as replacement")
else:
    print("The pm25 column exists or no suitable replacement was found")
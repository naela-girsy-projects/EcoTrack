# Save this as check_columns.py in your project root
import pandas as pd

# Load the generated demo data
data_path = "output/demo/sensor/air_quality_data.csv"
data = pd.read_csv(data_path)

# Print all column names
print("Available columns:")
for col in data.columns:
    print(f"- {col}")

# Print the first few rows to see the data structure
print("\nSample data:")
print(data.head())
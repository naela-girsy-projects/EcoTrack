import os
import sys
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the necessary modules
from visualization.visualization_dashboard import DataVisualizer

# Path to your data file
data_path = "data/raw/data_fixed.csv"

# Load data
print(f"Loading data from {data_path}")
if os.path.exists(data_path):
    data = pd.read_csv(data_path)
    print(f"Successfully loaded data: {data.shape}")
else:
    print(f"Data file not found at {data_path}")
    sys.exit(1)

# Create visualizer and dashboard
visualizer = DataVisualizer()
app = visualizer.create_dashboard()

# Print the method that will be called
print("\nAbout to call run_dashboard...")
print("Calling: app.run(port=8053, debug=True, host='0.0.0.0')")

# Run the dashboard
print("\nStarting dashboard on port 8053...")
print("The dashboard will be available at:")
print("  http://localhost:8053")

# Run the dashboard directly without any wrapper functions
app.run(port=8053, debug=True, host='0.0.0.0')
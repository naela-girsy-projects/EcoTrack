import os
import sys
import pandas as pd
import traceback

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Import EcoTrack modules
    from visualization.visualization_dashboard import DataVisualizer
    
    # Path to your data file
    data_path = "/Users/naelamacbookair/desktop backup/self projects/EcoTrack/data/raw/data.csv"
    
    # Load data
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    print(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
    print("Columns:", data.columns.tolist())
    
    # Create visualizer
    print("Creating data visualizer...")
    visualizer = DataVisualizer()
    
    # Create dashboard
    print("Creating dashboard application...")
    app = visualizer.create_dashboard()
    
    # Run dashboard with debug mode
    print("Starting dashboard on port 8060...")
    visualizer.run_dashboard(app, port=8060, debug=True)
    
except Exception as e:
    print("=" * 50)
    print("ERROR STARTING DASHBOARD:")
    print(str(e))
    print("\nFull traceback:")
    traceback.print_exc()
    print("=" * 50)
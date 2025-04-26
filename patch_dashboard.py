import os
import sys
import pandas as pd
import traceback

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Import EcoTrack dashboard module
    print("Importing visualization modules...")
    from visualization.visualization_dashboard import MapVisualizer, DataVisualizer

    # Load data
    print("Loading data...")
    data_path = "/Users/naelamacbookair/desktop backup/self projects/EcoTrack/data/raw/data_fixed.csv"
    data = pd.read_csv(data_path)
    print(f"Data loaded successfully: {data.shape}")
    
    # Print data columns for reference
    print("Columns:", data.columns.tolist())
    
    # Create a simplified dashboard
    print("Creating simplified dashboard...")
    visualizer = DataVisualizer()
    
    # Print the run_dashboard method to see its implementation
    print("\nCurrent run_dashboard method:")
    print(DataVisualizer.run_dashboard.__code__.co_code)
    
    # Monkey patch the run_dashboard method to use app.run
    print("\nModifying run_dashboard method...")
    original_method = DataVisualizer.run_dashboard
    
    def patched_run_dashboard(self, app, port=8050, debug=False):
        print(f"PATCHED: Running dashboard on port {port} with debug={debug}")
        # The key change is using run instead of run_server
        return app.run(port=port, debug=debug, host='0.0.0.0')
    
    DataVisualizer.run_dashboard = patched_run_dashboard
    print("Method patched!")
    
    # Create a dashboard app
    print("\nCreating dashboard app...")
    app = visualizer.create_dashboard()
    
    # Run the dashboard
    print("\nStarting dashboard on port 8052...")
    print("The dashboard will be available at:")
    print("  http://localhost:8052")
    print("  http://127.0.0.1:8052")
    print("Press Ctrl+C to stop the dashboard")
    
    # This should now use our patched method
    visualizer.run_dashboard(app, port=8052, debug=True)
    
except Exception as e:
    print("\n" + "=" * 50)
    print("ERROR:")
    print(str(e))
    print("\nTraceback:")
    traceback.print_exc()
    print("=" * 50)
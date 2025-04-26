import os
import sys
import pandas as pd
import dash
from dash import dcc, html
import traceback

# Set up a very simple dashboard to test
print("Creating a simple test dashboard...")
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Dashboard Test"),
    html.P("If you can see this, the dashboard is working!")
])

# Debug information
print("=" * 50)
print("DEBUG INFORMATION:")
print(f"Python version: {sys.version}")
print(f"Dash version: {dash.__version__}")
print("=" * 50)

# Run dashboard with verbose logging
print("Starting dashboard on port 8055...")
print("The dashboard will be available at:")
print("  http://localhost:8055")
print("  http://127.0.0.1:8055")
print("Press Ctrl+C to stop the dashboard")

# Use app.run instead of app.run_server for Dash 3.0+
app.run(debug=True, port=8055, host='0.0.0.0')
#!/usr/bin/env python
"""
EcoTrack - Environmental Data Analysis Platform
Main application script

This script demonstrates the full capabilities of the EcoTrack platform by
integrating satellite imagery analysis, sensor data, and visualizations.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rasterio
import datetime
import argparse

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import EcoTrack modules
from data_processing.data_processing_utils import SatelliteImageProcessor, SensorDataProcessor, DataIntegrator
from data_processing.deforestation_detection import DeforestationDetector
from data_processing.pollution_monitoring import PollutionAnalyzer
from visualization.visualization_dashboard import MapVisualizer, DataVisualizer, PlotlyVisualizer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='EcoTrack - Environmental Data Analysis Platform')
    
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'deforestation', 'pollution', 'dashboard'],
                        help='Operation mode')
    
    parser.add_argument('--satellite-dir', type=str, 
                        default='data/satellite',
                        help='Directory containing satellite imagery')
    
    parser.add_argument('--sensor-dir', type=str, 
                        default='data/sensor',
                        help='Directory containing sensor data')
    
    parser.add_argument('--output-dir', type=str, 
                        default='output',
                        help='Directory for output files')
    
    parser.add_argument('--before-image', type=str,
                        help='Path to "before" satellite image for deforestation analysis')
    
    parser.add_argument('--after-image', type=str,
                        help='Path to "after" satellite image for deforestation analysis')
    
    parser.add_argument('--sensor-file', type=str,
                        help='Path to sensor data file for pollution analysis')
    
    parser.add_argument('--pollutant-col', type=str,
                        default='pm25',
                        help='Column name for pollutant in sensor data')
    
    parser.add_argument('--dashboard-port', type=int,
                        default=8050,
                        help='Port for the interactive dashboard')
    
    return parser.parse_args()


def create_demo_data(output_dir):
    """Create demo data for testing and examples"""
    print("Creating demo data...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'satellite'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sensor'), exist_ok=True)
    
    # Create demo sensor data
    # This demo data simulates air quality measurements from multiple sensors
    sensor_data = []
    
    # Generate 2 years of daily data for 5 different sensor locations
    locations = [
        {'name': 'Forest Edge', 'latitude': 37.7749, 'longitude': -122.4194, 'location_type': 'forest'},
        {'name': 'City Center', 'latitude': 40.7128, 'longitude': -74.0060, 'location_type': 'urban'},
        {'name': 'Industrial Zone', 'latitude': 33.7490, 'longitude': -84.3880, 'location_type': 'industrial'},
        {'name': 'Agricultural Area', 'latitude': 36.7783, 'longitude': -119.4179, 'location_type': 'agricultural'},
        {'name': 'Suburban Area', 'latitude': 39.9526, 'longitude': -75.1652, 'location_type': 'suburban'}
    ]
    
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime(2022, 1, 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)  # For reproducibility
    
    for location in locations:
        # Base pollution levels depend on location type
        if location['location_type'] == 'urban':
            base_pm25 = 15
            base_pm10 = 30
            base_o3 = 35
            base_no2 = 25
        elif location['location_type'] == 'industrial':
            base_pm25 = 25
            base_pm10 = 50
            base_o3 = 40
            base_no2 = 35
        elif location['location_type'] == 'forest':
            base_pm25 = 5
            base_pm10 = 10
            base_o3 = 20
            base_no2 = 10
        elif location['location_type'] == 'agricultural':
            base_pm25 = 10
            base_pm10 = 20
            base_o3 = 30
            base_no2 = 15
        else:  # suburban
            base_pm25 = 8
            base_pm10 = 15
            base_o3 = 25
            base_no2 = 20
        
        # Add seasonal variations and random fluctuations
        for date in date_range:
            # Seasonal component (higher in winter for PM2.5 and PM10)
            day_of_year = date.dayofyear
            season_factor_pm = 1 + 0.5 * np.sin((day_of_year / 365) * 2 * np.pi + np.pi)  # Higher in winter
            season_factor_o3 = 1 + 0.7 * np.sin((day_of_year / 365) * 2 * np.pi)  # Higher in summer
            
            # Random daily fluctuation
            daily_fluctuation_pm25 = np.random.normal(0, 3)
            daily_fluctuation_pm10 = np.random.normal(0, 6)
            daily_fluctuation_o3 = np.random.normal(0, 5)
            daily_fluctuation_no2 = np.random.normal(0, 4)
            
            # Weekend effect (lower NO2 on weekends)
            weekend_factor = 0.7 if date.weekday() >= 5 else 1.0
            
            # Linear trend (improving air quality over time)
            days_elapsed = (date - start_date).days
            trend_factor = 1 - (days_elapsed / (end_date - start_date).days) * 0.2
            
            # Calculate final values
            pm25 = max(0, base_pm25 * season_factor_pm * trend_factor + daily_fluctuation_pm25)
            pm10 = max(0, base_pm10 * season_factor_pm * trend_factor + daily_fluctuation_pm10)
            o3 = max(0, base_o3 * season_factor_o3 * trend_factor + daily_fluctuation_o3)
            no2 = max(0, base_no2 * weekend_factor * trend_factor + daily_fluctuation_no2)
            
            # Calculate a composite air quality index
            aqi = (pm25 * 0.4 + pm10 * 0.2 + o3 * 0.2 + no2 * 0.2) * 2
            
            # Add to sensor data
            sensor_data.append({
                'timestamp': date,
                'sensor_id': f"sensor_{locations.index(location) + 1}",
                'location': location['name'],
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'location_type': location['location_type'],
                'pm25': round(pm25, 2),
                'pm10': round(pm10, 2),
                'o3': round(o3, 2),
                'no2': round(no2, 2),
                'temperature': round(20 + 15 * np.sin((day_of_year / 365) * 2 * np.pi) + np.random.normal(0, 3), 2),
                'humidity': round(50 + 20 * np.sin((day_of_year / 365) * 2 * np.pi + np.pi) + np.random.normal(0, 5), 2),
                'aqi': round(aqi, 2)
            })
    
    # Create DataFrame and save
    sensor_df = pd.DataFrame(sensor_data)
    sensor_csv_path = os.path.join(output_dir, 'sensor', 'air_quality_data.csv')
    sensor_df.to_csv(sensor_csv_path, index=False)
    print(f"Demo sensor data saved to {sensor_csv_path}")
    
    # We would normally create satellite imagery here, but this requires actual
    # satellite data or simulated raster files, which is beyond the scope of this script.
    # Instead, we'll just create a README file explaining how to get real data.
    
    readme_path = os.path.join(output_dir, 'satellite', 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("Demo Satellite Data\n")
        f.write("===================\n\n")
        f.write("For real satellite data, you can download Landsat, Sentinel, or MODIS data from:\n")
        f.write("- USGS Earth Explorer: https://earthexplorer.usgs.gov/\n")
        f.write("- Copernicus Open Access Hub: https://scihub.copernicus.eu/\n")
        f.write("- NASA Earthdata: https://earthdata.nasa.gov/\n\n")
        f.write("Place downloaded files in this directory for use with EcoTrack.\n")
    
    print(f"Created README for satellite data at {readme_path}")
    
    # Create a GeoJSON file with geographic regions for mapping
    regions = [
        {
            'name': 'Region A',
            'coordinates': [
                [-122.5, 37.7],
                [-122.4, 37.7],
                [-122.4, 37.8],
                [-122.5, 37.8],
                [-122.5, 37.7]
            ],
            'deforestation_rate': 5.2,
            'forest_coverage': 73.4,
            'risk_level': 'Medium'
        },
        {
            'name': 'Region B',
            'coordinates': [
                [-74.1, 40.7],
                [-74.0, 40.7],
                [-74.0, 40.8],
                [-74.1, 40.8],
                [-74.1, 40.7]
            ],
            'deforestation_rate': 8.7,
            'forest_coverage': 45.2,
            'risk_level': 'High'
        },
        {
            'name': 'Region C',
            'coordinates': [
                [-84.4, 33.7],
                [-84.3, 33.7],
                [-84.3, 33.8],
                [-84.4, 33.8],
                [-84.4, 33.7]
            ],
            'deforestation_rate': 2.1,
            'forest_coverage': 62.8,
            'risk_level': 'Low'
        },
        {
            'name': 'Region D',
            'coordinates': [
                [-119.5, 36.7],
                [-119.4, 36.7],
                [-119.4, 36.8],
                [-119.5, 36.8],
                [-119.5, 36.7]
            ],
            'deforestation_rate': 3.5,
            'forest_coverage': 58.9,
            'risk_level': 'Medium'
        },
        {
            'name': 'Region E',
            'coordinates': [
                [-75.2, 39.9],
                [-75.1, 39.9],
                [-75.1, 40.0],
                [-75.2, 40.0],
                [-75.2, 39.9]
            ],
            'deforestation_rate': 1.8,
            'forest_coverage': 82.3,
            'risk_level': 'Low'
        }
    ]
    
    # Create GeoDataFrame
    polygons = []
    attributes = []
    
    for region in regions:
        poly = Polygon(region['coordinates'])
        polygons.append(poly)
        
        attributes.append({
            'name': region['name'],
            'deforestation_rate': region['deforestation_rate'],
            'forest_coverage': region['forest_coverage'],
            'risk_level': region['risk_level']
        })
    
    gdf = gpd.GeoDataFrame(attributes, geometry=polygons, crs="EPSG:4326")
    
    # Save GeoJSON
    geojson_path = os.path.join(output_dir, 'regions.geojson')
    gdf.to_file(geojson_path, driver='GeoJSON')
    print(f"Demo geographic regions saved to {geojson_path}")
    
    return sensor_df, gdf


def analyze_deforestation(before_image_path, after_image_path, output_dir):
    """
    Analyze deforestation between two satellite images
    
    Parameters:
    -----------
    before_image_path : str
        Path to the earlier satellite image
    after_image_path : str
        Path to the later satellite image
    output_dir : str
        Directory to save outputs
    
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    print(f"Analyzing deforestation between {os.path.basename(before_image_path)} and {os.path.basename(after_image_path)}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize satellite image processor
    satellite_processor = SatelliteImageProcessor(os.path.dirname(before_image_path))
    
    # Initialize deforestation detector
    deforestation_detector = DeforestationDetector(os.path.join(output_dir, 'models'))
    
    # Load images
    before_img = satellite_processor.load_image(before_image_path)
    after_img = satellite_processor.load_image(after_image_path)
    
    if before_img is None or after_img is None:
        print("Error: Failed to load satellite images")
        return None
    
    print("Performing NDVI difference analysis...")
    # Detect deforestation using NDVI difference
    ndvi_mask, ndvi_percent = deforestation_detector.ndvi_difference(before_img, after_img, threshold=0.2)
    
    print("Performing K-means classification analysis...")
    # Detect deforestation using K-means classification
    kmeans_mask, kmeans_percent = deforestation_detector.detect_change_with_kmeans(before_img, after_img, n_clusters=5)
    
    print("Generating deforestation visualization...")
    # Create visualization
    fig = satellite_processor.visualize_deforestation(
        before_image_path, after_image_path, 
        output_path=os.path.join(output_dir, 'deforestation_visualization.png')
    )
    
    print("Extracting deforestation polygons...")
    # Convert deforestation mask to polygons
    polygons = deforestation_detector.get_deforestation_polygons(
        ndvi_mask, before_img.transform, before_img.crs
    )
    
    if polygons is not None:
        # Save polygons to GeoJSON
        polygons_path = os.path.join(output_dir, 'deforestation_polygons.geojson')
        polygons.to_file(polygons_path, driver='GeoJSON')
        print(f"Deforestation polygons saved to {polygons_path}")
        
        # Calculate total deforested area in km²
        total_deforested_area = polygons['area_km2'].sum()
    else:
        total_deforested_area = 0
    
    # Compile results
    results = {
        'ndvi_percent_deforested': ndvi_percent,
        'kmeans_percent_deforested': kmeans_percent,
        'total_deforested_area_km2': total_deforested_area,
        'deforestation_polygons': polygons,
        'visualization_path': os.path.join(output_dir, 'deforestation_visualization.png')
    }
    
    # Save results summary
    summary_path = os.path.join(output_dir, 'deforestation_results.txt')
    with open(summary_path, 'w') as f:
        f.write("Deforestation Analysis Results\n")
        f.write("==============================\n\n")
        f.write(f"Before image: {os.path.basename(before_image_path)}\n")
        f.write(f"After image: {os.path.basename(after_image_path)}\n\n")
        f.write(f"NDVI difference analysis:\n")
        f.write(f"  Deforested area: {ndvi_percent:.2f}%\n\n")
        f.write(f"K-means classification analysis:\n")
        f.write(f"  Deforested area: {kmeans_percent:.2f}%\n\n")
        f.write(f"Total deforested area: {total_deforested_area:.2f} km²\n")
    
    print(f"Results saved to {summary_path}")
    
    return results


#!/usr/bin/env python
"""
Comprehensive fix for the analyze_pollution function in main.py
Replace the existing analyze_pollution function with this one
"""

def analyze_pollution(sensor_file, pollutant_col, output_dir):
    """
    Analyze pollution trends from sensor data
    
    Parameters:
    -----------
    sensor_file : str
        Path to the sensor data file
    pollutant_col : str
        Column name for the pollutant to analyze
    output_dir : str
        Directory to save outputs
    
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    print(f"Analyzing pollution trends from {os.path.basename(sensor_file)}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize sensor data processor
    sensor_processor = SensorDataProcessor(os.path.dirname(sensor_file))
    
    # Initialize pollution analyzer
    pollution_analyzer = PollutionAnalyzer(os.path.join(output_dir, 'models'))
    
    # Load sensor data
    sensor_data = sensor_processor.load_data(sensor_file)
    
    if sensor_data is None:
        print("Error: Failed to load sensor data")
        return None
    
    print("Cleaning sensor data...")
    # Clean sensor data
    cleaned_data = sensor_processor.clean_sensor_data(sensor_data)
    
    # Ensure pollutant_col exists in the data
    if pollutant_col not in cleaned_data.columns:
        print(f"Warning: '{pollutant_col}' column not found in data")
        # Check for alternative column names (e.g., PM2.5 instead of pm25)
        alt_cols = [col for col in cleaned_data.columns if col.lower() == pollutant_col.lower()]
        if alt_cols:
            print(f"Using alternative column '{alt_cols[0]}' instead")
            # Create a copy with the expected name
            cleaned_data[pollutant_col] = cleaned_data[alt_cols[0]]
    
    print("Calculating pollution index...")
    # Calculate pollution index if multiple pollutants are available
    possible_pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
    available_pollutants = [col for col in possible_pollutants if col in cleaned_data.columns]
    
    if len(available_pollutants) > 1:
        pollution_index = sensor_processor.calculate_pollution_index(cleaned_data, available_pollutants)
        cleaned_data['pollution_index'] = pollution_index
    
    print("Aggregating data by time...")
    # Aggregate data by day
    daily_data = sensor_processor.aggregate_by_time(
        cleaned_data, time_column='timestamp', freq='1D'
    )
    
    # CRITICAL FIX: Add the pollutant_col column to daily_data based on mean
    mean_col = f"{pollutant_col}_mean"
    if pollutant_col not in daily_data.columns and mean_col in daily_data.columns:
        print(f"Adding '{pollutant_col}' column to daily data based on '{mean_col}'")
        daily_data[pollutant_col] = daily_data[mean_col]
    
    # Double-check that pollutant_col exists
    if pollutant_col not in daily_data.columns:
        print(f"ERROR: '{pollutant_col}' column still not found in daily data")
        print(f"Available columns: {daily_data.columns.tolist()}")
        # Use an available column instead
        if mean_col in daily_data.columns:
            print(f"Using '{mean_col}' for visualizations")
            # Rename the column temporarily for this analysis
            daily_data = daily_data.rename(columns={mean_col: pollutant_col})
    
    # Ensure timestamp is datetime type
    if 'timestamp' in daily_data.columns and not pd.api.types.is_datetime64_dtype(daily_data['timestamp']):
        daily_data['timestamp'] = pd.to_datetime(daily_data['timestamp'])
    
    # Aggregate data by month
    monthly_data = sensor_processor.aggregate_by_time(
        cleaned_data, time_column='timestamp', freq='1M'
    )
    
    print("Analyzing pollution trends...")
    # Analyze trends for all available pollutants
    trends = pollution_analyzer.analyze_pollution_trends(
        cleaned_data, available_pollutants, time_column='timestamp', freq='M'
    )
    
    print("Performing correlation analysis...")
    # Correlation analysis
    numerical_cols = cleaned_data.select_dtypes(include=[np.number]).columns
    corr_matrix = pollution_analyzer.correlation_analysis(
        cleaned_data, columns=numerical_cols, time_column='timestamp'
    )
    
    print("Training forecasting model...")
    # Train ARIMA model for forecasting
    target_col = pollutant_col if pollutant_col in cleaned_data.columns else available_pollutants[0]
    arima_model = pollution_analyzer.train_arima_model(
        cleaned_data, target_column=target_col, time_column='timestamp',
        order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)
    )
    
    # Generate forecast for the next 30 days
    forecast = pollution_analyzer.forecast_with_arima(
        arima_model, steps=30, confidence_interval=0.95
    )
    
    print("Creating visualizations...")
    # Create visualizations
    visualizer = DataVisualizer()
    
    # Verify visualization data
    print(f"Visualization data check - daily_data shape: {daily_data.shape}")
    print(f"Columns for visualization: {daily_data.columns.tolist()}")
    print(f"Using column '{pollutant_col}' for time series visualization")
    
    # Time series plot
    try:
        time_series_fig = visualizer.plot_time_series(
            daily_data, y_col=pollutant_col, x_col='timestamp',
            title=f'{pollutant_col.upper()} Levels Over Time',
            ylabel=f'{pollutant_col.upper()} Concentration',
            rolling_window=7, show_trend=True,
            output_path=os.path.join(output_dir, f'{pollutant_col}_time_series.png')
        )
        print(f"Time series plot saved to {os.path.join(output_dir, f'{pollutant_col}_time_series.png')}")
    except Exception as e:
        print(f"Error creating time series plot: {e}")
    
    # Correlation heatmap
    try:
        heatmap_fig = visualizer.plot_heatmap(
            corr_matrix,
            title='Correlation Between Environmental Variables',
            mask_upper=True,
            output_path=os.path.join(output_dir, 'correlation_heatmap.png')
        )
        print(f"Correlation heatmap saved to {os.path.join(output_dir, 'correlation_heatmap.png')}")
    except Exception as e:
        print(f"Error creating heatmap: {e}")
    
    # Forecast visualization
    try:
        if 'timestamp' in daily_data.columns:
            forecast_fig = pollution_analyzer.visualize_forecast(
                daily_data, forecast, target_column=pollutant_col,
                title=f'{pollutant_col.upper()} Forecast (Next 30 Days)',
                output_path=os.path.join(output_dir, f'{pollutant_col}_forecast.png')
            )
            print(f"Forecast visualization saved to {os.path.join(output_dir, f'{pollutant_col}_forecast.png')}")
    except Exception as e:
        print(f"Error creating forecast visualization: {e}")
    
    # Save trends data
    trends.to_csv(os.path.join(output_dir, 'pollution_trends.csv'))
    
    # Save correlation matrix
    corr_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
    
    # Save forecast data
    forecast.to_csv(os.path.join(output_dir, 'pollution_forecast.csv'))
    
    # Compile results
    results = {
        'daily_data': daily_data,
        'monthly_data': monthly_data,
        'trends': trends,
        'correlation_matrix': corr_matrix,
        'forecast': forecast,
        'time_series_path': os.path.join(output_dir, f'{pollutant_col}_time_series.png'),
        'heatmap_path': os.path.join(output_dir, 'correlation_heatmap.png'),
        'forecast_path': os.path.join(output_dir, f'{pollutant_col}_forecast.png')
    }
    
    # Save results summary
    summary_path = os.path.join(output_dir, 'pollution_results.txt')
    with open(summary_path, 'w') as f:
        f.write("Pollution Analysis Results\n")
        f.write("=========================\n\n")
        f.write(f"Data source: {os.path.basename(sensor_file)}\n")
        f.write(f"Target pollutant: {target_col}\n\n")
        
        f.write("Summary Statistics:\n")
        stats = cleaned_data[target_col].describe()
        for stat_name, stat_value in stats.items():
            f.write(f"  {stat_name}: {stat_value:.2f}\n")
        
        f.write("\nTrend Analysis:\n")
        if f'{target_col}_trend' in trends.columns:
            trend_value = trends[f'{target_col}_trend'].iloc[0]
            trend_direction = "increasing" if trend_value > 0 else "decreasing"
            f.write(f"  {target_col} is {trend_direction} at a rate of {abs(trend_value):.4f} units per month\n")
        
        f.write("\nForecast (30-day average):\n")
        forecast_avg = forecast['forecast'].mean()
        f.write(f"  Predicted average {target_col}: {forecast_avg:.2f}\n")
    
    print(f"Results saved to {summary_path}")
    
    return results
"""
EcoTrack - Data Processing Utilities
This module contains utility functions for processing different types of environmental data.
"""

import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.plot import show
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import cv2
from sklearn.preprocessing import StandardScaler
import earthpy.spatial as es
import earthpy.plot as ep

class SatelliteImageProcessor:
    """Class for processing satellite imagery data"""
    
    def __init__(self, data_dir):
        """
        Initialize the satellite image processor
        
        Parameters:
        -----------
        data_dir : str
            Directory containing satellite image data
        """
        self.data_dir = data_dir
        self.image_cache = {}
        
    def load_image(self, image_path):
        """
        Load satellite image using rasterio
        
        Parameters:
        -----------
        image_path : str
            Path to the satellite image file
            
        Returns:
        --------
        rasterio.DatasetReader
            Loaded satellite image
        """
        if image_path in self.image_cache:
            return self.image_cache[image_path]
        
        try:
            img = rasterio.open(image_path)
            self.image_cache[image_path] = img
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def calculate_ndvi(self, image):
        """
        Calculate Normalized Difference Vegetation Index (NDVI)
        
        Parameters:
        -----------
        image : rasterio.DatasetReader
            Satellite image with NIR and RED bands
            
        Returns:
        --------
        numpy.ndarray
            NDVI values ranging from -1 to 1
        """
        # Get band indices (assuming band ordering: RED=1, NIR=4 for Landsat)
        red_band = 1
        nir_band = 4
        
        # Read the bands as arrays
        red = image.read(red_band).astype(float)
        nir = image.read(nir_band).astype(float)
        
        # Calculate NDVI
        # Avoid division by zero
        denominator = nir + red
        ndvi = np.where(
            denominator > 0,
            (nir - red) / denominator,
            0
        )
        
        return ndvi
    
    def detect_deforestation(self, before_img_path, after_img_path, threshold=0.2):
        """
        Detect deforestation by comparing NDVI values between two time periods
        
        Parameters:
        -----------
        before_img_path : str
            Path to the earlier satellite image
        after_img_path : str
            Path to the later satellite image
        threshold : float
            NDVI difference threshold to classify as deforestation
            
        Returns:
        --------
        tuple
            (deforestation_mask, percent_deforested)
        """
        # Load images
        before_img = self.load_image(before_img_path)
        after_img = self.load_image(after_img_path)
        
        if before_img is None or after_img is None:
            return None, 0
        
        # Calculate NDVI for both images
        before_ndvi = self.calculate_ndvi(before_img)
        after_ndvi = self.calculate_ndvi(after_img)
        
        # Calculate difference
        ndvi_diff = before_ndvi - after_ndvi
        
        # Create deforestation mask (areas where NDVI decreased significantly)
        deforestation_mask = ndvi_diff > threshold
        
        # Calculate percentage of deforested area
        percent_deforested = (np.sum(deforestation_mask) / deforestation_mask.size) * 100
        
        return deforestation_mask, percent_deforested
    
    def visualize_deforestation(self, before_img_path, after_img_path, output_path=None, threshold=0.2):
        """
        Create visualization of deforestation between two time periods
        
        Parameters:
        -----------
        before_img_path : str
            Path to the earlier satellite image
        after_img_path : str
            Path to the later satellite image
        output_path : str
            Path to save the visualization
        threshold : float
            NDVI difference threshold to classify as deforestation
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with deforestation visualization
        """
        # Detect deforestation
        deforestation_mask, percent = self.detect_deforestation(before_img_path, after_img_path, threshold)
        
        if deforestation_mask is None:
            return None
        
        # Get RGB data for visualization
        after_img = self.load_image(after_img_path)
        rgb_bands = [after_img.read(i) for i in [3, 2, 1]]  # Assuming RGB = bands 3,2,1
        rgb = np.dstack(rgb_bands)
        
        # Normalize for visualization
        rgb_norm = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Show RGB image
        ax.imshow(rgb_norm)
        
        # Overlay deforestation mask in red with transparency
        deforestation_overlay = np.zeros_like(rgb_norm)
        deforestation_overlay[deforestation_mask] = [255, 0, 0]  # Red for deforested areas
        ax.imshow(deforestation_overlay, alpha=0.4)
        
        ax.set_title(f"Deforestation Analysis (Red areas)\nDeforested area: {percent:.2f}%")
        ax.set_axis_off()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig

class SensorDataProcessor:
    """Class for processing environmental sensor data"""
    
    def __init__(self, data_dir):
        """
        Initialize the sensor data processor
        
        Parameters:
        -----------
        data_dir : str
            Directory containing sensor data files
        """
        self.data_dir = data_dir
        self.data_cache = {}
        
    def load_data(self, file_path, force_reload=False):
        """
        Load sensor data from CSV or JSON file
        
        Parameters:
        -----------
        file_path : str
            Path to the sensor data file
        force_reload : bool
            Whether to force reload data from file even if cached
            
        Returns:
        --------
        pandas.DataFrame
            Loaded sensor data
        """
        if file_path in self.data_cache and not force_reload:
            return self.data_cache[file_path]
        
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = pd.DataFrame(json.load(f))
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Convert timestamp to datetime if it exists
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            self.data_cache[file_path] = data
            return data
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return None
    
    def clean_sensor_data(self, data):
        """
        Clean sensor data by handling missing values, outliers, etc.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw sensor data
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned sensor data
        """
        if data is None:
            return None
        
        # Create a copy of the data
        cleaned = data.copy()
        
        # Handle missing values
        numerical_cols = cleaned.select_dtypes(include=[np.number]).columns
        categorical_cols = cleaned.select_dtypes(include=['object']).columns
        
        # Fill numerical missing values with column median
        for col in numerical_cols:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        
        # Fill categorical missing values with most common value
        for col in categorical_cols:
            most_common = cleaned[col].mode()[0]
            cleaned[col] = cleaned[col].fillna(most_common)
        
        # Handle outliers using IQR method for numerical columns
        for col in numerical_cols:
            Q1 = cleaned[col].quantile(0.25)
            Q3 = cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers with bounds
            cleaned[col] = np.where(cleaned[col] < lower_bound, lower_bound, cleaned[col])
            cleaned[col] = np.where(cleaned[col] > upper_bound, upper_bound, cleaned[col])
        
        return cleaned
    
    def aggregate_by_time(self, data, time_column='timestamp', freq='1D', agg_funcs=None):
        """
        Aggregate sensor data by time period
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Sensor data
        time_column : str
            Column containing timestamp data
        freq : str
            Frequency for aggregation (e.g., '1H' for hourly, '1D' for daily)
        agg_funcs : dict
            Dictionary mapping column names to aggregation functions
            
        Returns:
        --------
        pandas.DataFrame
            Aggregated sensor data
        """
        if data is None:
            return None
        
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_dtype(data[time_column]):
            data[time_column] = pd.to_datetime(data[time_column])
        
        # Set default aggregation functions if none provided
        if agg_funcs is None:
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            agg_funcs = {col: ['mean', 'min', 'max', 'std'] for col in numerical_cols if col != time_column}
        
        # Set time column as index
        data_indexed = data.set_index(time_column)
        
        # Perform aggregation
        aggregated = data_indexed.groupby(pd.Grouper(freq=freq)).agg(agg_funcs)
        
        # Flatten multi-index columns
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
        
        # Reset index to make timestamp a column again
        aggregated.reset_index(inplace=True)
        
        return aggregated
    
    def calculate_pollution_index(self, data, pollutants=None, weights=None):
        """
        Calculate a composite pollution index from multiple pollutant measurements
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Sensor data with pollutant measurements
        pollutants : list
            List of column names for pollutant measurements
        weights : dict
            Dictionary mapping pollutant names to their weights in the index
            
        Returns:
        --------
        pandas.Series
            Calculated pollution index
        """
        if data is None:
            return None
        
        # Default pollutants if none specified
        if pollutants is None:
            # Common air pollutants
            possible_pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO']
            pollutants = [col for col in possible_pollutants if col in data.columns]
            
            if not pollutants:
                print("No recognized pollutant columns found in data")
                return None
        
        # Default weights if none specified (equal weighting)
        if weights is None:
            weights = {pollutant: 1.0/len(pollutants) for pollutant in pollutants}
        
        # Normalize each pollutant to 0-1 scale
        scaler = StandardScaler()
        normalized_data = pd.DataFrame(
            scaler.fit_transform(data[pollutants]),
            columns=pollutants
        )
        
        # Calculate weighted sum
        pollution_index = pd.Series(0, index=data.index)
        for pollutant in pollutants:
            if pollutant in weights:
                pollution_index += normalized_data[pollutant] * weights[pollutant]
        
        # Scale to 0-100 for interpretability
        min_val = pollution_index.min()
        max_val = pollution_index.max()
        pollution_index = 100 * (pollution_index - min_val) / (max_val - min_val)
        
        return pollution_index

class DataIntegrator:
    """Class for integrating different data sources"""
    
    def __init__(self, satellite_processor, sensor_processor):
        """
        Initialize the data integrator
        
        Parameters:
        -----------
        satellite_processor : SatelliteImageProcessor
            Processor for satellite data
        sensor_processor : SensorDataProcessor
            Processor for sensor data
        """
        self.satellite_processor = satellite_processor
        self.sensor_processor = sensor_processor
    
    def integrate_by_location(self, satellite_data, sensor_data, location_col='location'):
        """
        Integrate satellite and sensor data based on location
        
        Parameters:
        -----------
        satellite_data : geopandas.GeoDataFrame
            Satellite-derived data with geometry
        sensor_data : pandas.DataFrame
            Sensor data with location information
        location_col : str
            Column in sensor_data containing location information
            
        Returns:
        --------
        geopandas.GeoDataFrame
            Integrated data
        """
        # Convert sensor data to GeoDataFrame if it's not already
        if not isinstance(sensor_data, gpd.GeoDataFrame):
            # Check if we have lat/lon columns
            if 'latitude' in sensor_data.columns and 'longitude' in sensor_data.columns:
                # Create geometry from lat/lon
                geometry = gpd.points_from_xy(sensor_data.longitude, sensor_data.latitude)
                sensor_gdf = gpd.GeoDataFrame(sensor_data, geometry=geometry, crs="EPSG:4326")
            elif location_col in sensor_data.columns:
                # Try to geocode location names
                from geopy.geocoders import Nominatim
                geolocator = Nominatim(user_agent="ecotrack")
                
                # Create a function to get coordinates from location name
                def get_coords(location_name):
                    try:
                        location = geolocator.geocode(location_name)
                        if location:
                            return (location.longitude, location.latitude)
                        return (np.nan, np.nan)
                    except:
                        return (np.nan, np.nan)
                
                # Apply geocoding (warning: this can be slow for large datasets)
                coords = sensor_data[location_col].apply(get_coords)
                sensor_data['longitude'] = coords.apply(lambda x: x[0])
                sensor_data['latitude'] = coords.apply(lambda x: x[1])
                
                # Drop rows with failed geocoding
                sensor_data = sensor_data.dropna(subset=['longitude', 'latitude'])
                
                # Create GeoDataFrame
                geometry = gpd.points_from_xy(sensor_data.longitude, sensor_data.latitude)
                sensor_gdf = gpd.GeoDataFrame(sensor_data, geometry=geometry, crs="EPSG:4326")
            else:
                print("Cannot create GeoDataFrame: No location information found")
                return None
        else:
            sensor_gdf = sensor_data
        
        # Ensure both GeoDataFrames have the same CRS
        if satellite_data.crs != sensor_gdf.crs:
            sensor_gdf = sensor_gdf.to_crs(satellite_data.crs)
        
        # Perform spatial join
        joined_data = gpd.sjoin(sensor_gdf, satellite_data, how="left", op="within")
        
        return joined_data
    
    def integrate_by_time(self, satellite_data, sensor_data, satellite_time_col='date', sensor_time_col='timestamp', max_time_diff='7D'):
        """
        Integrate satellite and sensor data based on time proximity
        
        Parameters:
        -----------
        satellite_data : pandas.DataFrame
            Satellite-derived data with time information
        sensor_data : pandas.DataFrame
            Sensor data with time information
        satellite_time_col : str
            Column in satellite_data containing time information
        sensor_time_col : str
            Column in sensor_data containing time information
        max_time_diff : str or timedelta
            Maximum time difference for matching records
            
        Returns:
        --------
        pandas.DataFrame
            Integrated data
        """
        # Ensure time columns are datetime
        if not pd.api.types.is_datetime64_dtype(satellite_data[satellite_time_col]):
            satellite_data[satellite_time_col] = pd.to_datetime(satellite_data[satellite_time_col])
        
        if not pd.api.types.is_datetime64_dtype(sensor_data[sensor_time_col]):
            sensor_data[sensor_time_col] = pd.to_datetime(sensor_data[sensor_time_col])
        
        # Convert max_time_diff to timedelta if it's a string
        if isinstance(max_time_diff, str):
            max_time_diff = pd.Timedelta(max_time_diff)
        
        # Initialize result DataFrame
        result = []
        
        # For each sensor data point, find the closest satellite data point in time
        for _, sensor_row in sensor_data.iterrows():
            sensor_time = sensor_row[sensor_time_col]
            
            # Calculate time difference
            satellite_data['time_diff'] = abs(satellite_data[satellite_time_col] - sensor_time)
            
            # Find the closest record within max_time_diff
            closest = satellite_data[satellite_data['time_diff'] <= max_time_diff].sort_values('time_diff').iloc[0:1]
            
            if len(closest) > 0:
                # Combine the sensor data with closest satellite data
                combined_row = pd.concat([sensor_row, closest.iloc[0].drop(columns=['time_diff'])])
                result.append(combined_row)
        
        if result:
            return pd.DataFrame(result)
        else:
            print("No matching records found within the specified time difference")
            return None
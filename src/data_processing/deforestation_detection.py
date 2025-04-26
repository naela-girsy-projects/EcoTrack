"""
EcoTrack - Deforestation Detection Algorithms
This module contains algorithms for detecting deforestation from satellite imagery.
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import geopandas as gpd
from shapely.geometry import Point, Polygon
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class DeforestationDetector:
    """Class implementing various algorithms for deforestation detection"""
    
    def __init__(self, model_dir=None):
        """
        Initialize the deforestation detector
        
        Parameters:
        -----------
        model_dir : str
            Directory to save/load trained models
        """
        self.model_dir = model_dir
        self.models = {}
        
        # Create model directory if it doesn't exist
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def ndvi_difference(self, before_img, after_img, threshold=0.2):
        """
        Detect deforestation using NDVI difference between two time periods
        
        Parameters:
        -----------
        before_img : rasterio.DatasetReader
            Earlier satellite image
        after_img : rasterio.DatasetReader
            Later satellite image
        threshold : float
            NDVI difference threshold to classify as deforestation
            
        Returns:
        --------
        tuple
            (deforestation_mask, percent_deforested)
        """
        # Calculate NDVI for both images (assuming band order where NIR=4, RED=1)
        nir_band_idx, red_band_idx = 4, 1
        
        # Read the bands as arrays
        before_red = before_img.read(red_band_idx).astype(float)
        before_nir = before_img.read(nir_band_idx).astype(float)
        after_red = after_img.read(red_band_idx).astype(float)
        after_nir = after_img.read(nir_band_idx).astype(float)
        
        # Calculate NDVI for before image
        denominator = before_nir + before_red
        before_ndvi = np.where(
            denominator > 0,
            (before_nir - before_red) / denominator,
            0
        )
        
        # Calculate NDVI for after image
        denominator = after_nir + after_red
        after_ndvi = np.where(
            denominator > 0,
            (after_nir - after_red) / denominator,
            0
        )
        
        # Calculate difference
        ndvi_diff = before_ndvi - after_ndvi
        
        # Create deforestation mask (areas where NDVI decreased significantly)
        deforestation_mask = ndvi_diff > threshold
        
        # Calculate percentage of deforested area
        percent_deforested = (np.sum(deforestation_mask) / deforestation_mask.size) * 100
        
        return deforestation_mask, percent_deforested
    
    def kmeans_classification(self, img, n_clusters=5):
        """
        Classify land cover types using K-means clustering
        
        Parameters:
        -----------
        img : rasterio.DatasetReader
            Satellite image
        n_clusters : int
            Number of clusters for K-means
            
        Returns:
        --------
        numpy.ndarray
            Classified image with cluster labels
        """
        # Read all bands and reshape for K-means
        bands = [img.read(i) for i in range(1, img.count + 1)]
        data = np.dstack(bands)
        rows, cols, bands = data.shape
        data_reshaped = data.reshape((rows * cols, bands))
        
        # Remove NaN values
        valid_pixels = ~np.isnan(data_reshaped).any(axis=1)
        data_valid = data_reshaped[valid_pixels]
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data_valid)
        
        # Reconstruct classified image
        classified = np.zeros(rows * cols, dtype=np.uint8)
        classified[valid_pixels] = labels
        classified = classified.reshape((rows, cols))
        
        return classified
    
    def detect_change_with_kmeans(self, before_img, after_img, n_clusters=5, forest_clusters=None):
        """
        Detect land cover change using K-means classification
        
        Parameters:
        -----------
        before_img : rasterio.DatasetReader
            Earlier satellite image
        after_img : rasterio.DatasetReader
            Later satellite image
        n_clusters : int
            Number of clusters for K-means
        forest_clusters : list
            List of cluster IDs that represent forest
            
        Returns:
        --------
        tuple
            (deforestation_mask, percent_deforested)
        """
        # Classify both images
        before_classified = self.kmeans_classification(before_img, n_clusters)
        after_classified = self.kmeans_classification(after_img, n_clusters)
        
        # If forest clusters not provided, use NDVI to determine them
        if forest_clusters is None:
            # Calculate NDVI
            red_band_idx, nir_band_idx = 1, 4
            red = before_img.read(red_band_idx).astype(float)
            nir = before_img.read(nir_band_idx).astype(float)
            
            denominator = nir + red
            ndvi = np.where(
                denominator > 0,
                (nir - red) / denominator,
                0
            )
            
            # Calculate average NDVI for each cluster
            cluster_ndvi = {}
            for i in range(n_clusters):
                mask = before_classified == i
                if np.sum(mask) > 0:
                    cluster_ndvi[i] = np.mean(ndvi[mask])
            
            # Sort clusters by NDVI (higher is more vegetation)
            sorted_clusters = sorted(cluster_ndvi.items(), key=lambda x: x[1], reverse=True)
            
            # Take top clusters as forest (assuming top 40% of clusters are forest)
            num_forest_clusters = max(1, int(n_clusters * 0.4))
            forest_clusters = [cluster for cluster, _ in sorted_clusters[:num_forest_clusters]]
        
        # Create forest masks
        before_forest = np.isin(before_classified, forest_clusters)
        after_forest = np.isin(after_classified, forest_clusters)
        
        # Detect deforestation (was forest, now isn't)
        deforestation_mask = np.logical_and(before_forest, ~after_forest)
        
        # Calculate percentage
        percent_deforested = (np.sum(deforestation_mask) / np.sum(before_forest)) * 100 if np.sum(before_forest) > 0 else 0
        
        return deforestation_mask, percent_deforested
    
    def train_random_forest(self, training_data, features, labels):
        """
        Train a Random Forest classifier for land cover classification
        
        Parameters:
        -----------
        training_data : pandas.DataFrame
            Training data with features and labels
        features : list
            List of feature column names
        labels : str
            Column name for labels
            
        Returns:
        --------
        sklearn.ensemble.RandomForestClassifier
            Trained Random Forest model
        """
        # Create and train the model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(training_data[features], training_data[labels])
        
        # Save model
        if self.model_dir:
            import joblib
            model_path = os.path.join(self.model_dir, 'random_forest_model.joblib')
            joblib.dump(rf, model_path)
            print(f"Model saved to {model_path}")
        
        self.models['random_forest'] = rf
        return rf
    
    def predict_with_random_forest(self, img, rf_model=None, features=None):
        """
        Classify land cover using a trained Random Forest model
        
        Parameters:
        -----------
        img : rasterio.DatasetReader
            Satellite image
        rf_model : sklearn.ensemble.RandomForestClassifier
            Trained Random Forest model (if None, uses stored model)
        features : list
            List of features (band indices) to use
            
        Returns:
        --------
        numpy.ndarray
            Classified image with predicted labels
        """
        # Use stored model if none provided
        if rf_model is None:
            if 'random_forest' in self.models:
                rf_model = self.models['random_forest']
            else:
                print("No Random Forest model available. Please train or provide a model.")
                return None
        
        # Default to using all bands as features
        if features is None:
            features = list(range(1, img.count + 1))
        
        # Read specified bands
        band_data = [img.read(i) for i in features]
        data = np.dstack(band_data)
        
        # Reshape for prediction
        rows, cols, bands = data.shape
        data_reshaped = data.reshape((rows * cols, bands))
        
        # Remove NaN values
        valid_pixels = ~np.isnan(data_reshaped).any(axis=1)
        data_valid = data_reshaped[valid_pixels]
        
        # Predict
        predictions = rf_model.predict(data_valid)
        
        # Reconstruct classified image
        classified = np.zeros(rows * cols, dtype=np.uint8)
        classified[valid_pixels] = predictions
        classified = classified.reshape((rows, cols))
        
        return classified
    
    def build_cnn_model(self, input_shape, num_classes):
        """
        Build a Convolutional Neural Network for image classification
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input images (height, width, channels)
        num_classes : int
            Number of output classes
            
        Returns:
        --------
        tensorflow.keras.models.Sequential
            Compiled CNN model
        """
        # Create a sequential model
        model = Sequential()
        
        # Add convolutional layers
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        # Flatten and add dense layers
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_cnn(self, train_data, train_labels, validation_data=None, epochs=20, batch_size=32):
        """
        Train a CNN model for land cover classification
        
        Parameters:
        -----------
        train_data : numpy.ndarray
            Training image patches
        train_labels : numpy.ndarray
            Training labels
        validation_data : tuple
            (val_data, val_labels) for validation
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        tensorflow.keras.models.Sequential
            Trained CNN model
        """
        # Get input shape and number of classes
        input_shape = train_data.shape[1:]
        num_classes = len(np.unique(train_labels))
        
        # Build the model
        model = self.build_cnn_model(input_shape, num_classes)
        
        # Train the model
        history = model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )
        
        # Save the model
        if self.model_dir:
            model_path = os.path.join(self.model_dir, 'cnn_model.h5')
            model.save(model_path)
            print(f"Model saved to {model_path}")
        
        self.models['cnn'] = model
        return model, history
    
    def predict_with_cnn(self, img, model=None, patch_size=64, stride=32):
        """
        Classify satellite image using a trained CNN model
        
        Parameters:
        -----------
        img : rasterio.DatasetReader
            Satellite image
        model : tensorflow.keras.models.Model
            Trained CNN model
        patch_size : int
            Size of image patches for classification
        stride : int
            Stride for moving window
            
        Returns:
        --------
        numpy.ndarray
            Classified image
        """
        # Use stored model if none provided
        if model is None:
            if 'cnn' in self.models:
                model = self.models['cnn']
            else:
                print("No CNN model available. Please train or provide a model.")
                return None
        
        # Read the image data
        image_data = np.dstack([img.read(i) for i in range(1, img.count + 1)])
        height, width, channels = image_data.shape
        
        # Pad the image to ensure full coverage
        pad_h = (patch_size - height % patch_size) % patch_size
        pad_w = (patch_size - width % patch_size) % patch_size
        padded_image = np.pad(image_data, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        
        # Create a result array
        padded_height, padded_width = padded_image.shape[:2]
        result = np.zeros((padded_height, padded_width), dtype=np.uint8)
        counts = np.zeros((padded_height, padded_width), dtype=np.uint8)
        
        # Extract patches and classify
        for y in range(0, padded_height - patch_size + 1, stride):
            for x in range(0, padded_width - patch_size + 1, stride):
                # Extract patch
                patch = padded_image[y:y+patch_size, x:x+patch_size]
                
                # Skip if patch has NaN values
                if np.isnan(patch).any():
                    continue
                
                # Normalize patch
                patch = patch / 255.0
                
                # Predict
                pred = model.predict(np.expand_dims(patch, axis=0))[0]
                pred_class = np.argmax(pred)
                
                # Add to result
                result[y:y+patch_size, x:x+patch_size] += pred_class
                counts[y:y+patch_size, x:x+patch_size] += 1
        
        # Average the predictions (handling divide by zero)
        counts = np.where(counts == 0, 1, counts)
        final_result = (result / counts).astype(np.uint8)
        
        # Crop back to original size
        final_result = final_result[:height, :width]
        
        return final_result
    
    def get_deforestation_polygons(self, deforestation_mask, transform, crs, min_area=100):
        """
        Convert a deforestation mask to vector polygons
        
        Parameters:
        -----------
        deforestation_mask : numpy.ndarray
            Binary mask of deforested areas
        transform : affine.Affine
            Transformation from pixel to geographic coordinates
        crs : str or dict
            Coordinate reference system
        min_area : float
            Minimum area (in pixels) for polygons
            
        Returns:
        --------
        geopandas.GeoDataFrame
            GeoDataFrame with deforestation polygons
        """
        # Convert mask to uint8 for contour detection
        mask_uint8 = deforestation_mask.astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area and convert to polygons
        polygons = []
        areas = []
        
        for contour in contours:
            # Calculate area in pixels
            area = cv2.contourArea(contour)
            
            if area >= min_area:
                # Convert contour to polygon
                coords = []
                for point in contour.reshape(-1, 2):
                    # Convert pixel coordinates to geographic coordinates
                    x, y = point[0], point[1]
                    lon, lat = rasterio.transform.xy(transform, y, x)
                    coords.append((lon, lat))
                
                # Create polygon
                if len(coords) >= 3:  # Need at least 3 points for a polygon
                    poly = Polygon(coords)
                    polygons.append(poly)
                    areas.append(area)
        
        # Create GeoDataFrame
        if polygons:
            gdf = gpd.GeoDataFrame(
                {'area_pixels': areas, 'geometry': polygons},
                crs=crs
            )
            
            # Calculate area in square meters or km2
            gdf['area_m2'] = gdf.to_crs(epsg=3857).area
            gdf['area_km2'] = gdf['area_m2'] / 1_000_000
            
            return gdf
        else:
            print("No polygons found that meet the minimum area criteria")
            return gpd.GeoDataFrame(columns=['area_pixels', 'area_m2', 'area_km2', 'geometry'], geometry='geometry', crs=crs)
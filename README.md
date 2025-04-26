# EcoTrack: Environmental Data Analysis Platform

EcoTrack is a comprehensive data analytics platform for environmental monitoring that combines satellite imagery analysis with sensor data to track deforestation and pollution trends.

## Features

- **Satellite Imagery Analysis**: Detects changes in forest cover using various algorithms (NDVI difference, K-means classification)
- **Pollution Monitoring**: Analyzes sensor data to identify pollution trends and forecast future conditions
- **Data Integration**: Combines multiple data sources (satellite imagery, weather data, sensor readings)
- **Interactive Visualizations**: Creates maps, charts, and dashboards to visualize environmental trends
- **Predictive Models**: Forecasts potential environmental risks using time series analysis and machine learning

## Project Structure

```
EcoTrack/
├── data/                  # Data directory
│   ├── raw/               # Raw data files
│   ├── processed/         # Processed data files
│   ├── satellite/         # Satellite imagery
│   └── sensor/            # Sensor data
├── models/                # Trained models
├── notebooks/             # Jupyter notebooks for exploration and demos
├── src/                   # Source code
│   ├── data_processing/   # Data processing modules
│   ├── visualization/     # Visualization modules
│   ├── models/            # Model training and inference
│   └── api/               # API for web services
├── dashboard/             # Dashboard components
├── output/                # Output files and results
├── docs/                  # Documentation
└── main.py                # Main application script
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ecotrack.git
   cd ecotrack
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv ecotrack-env
   source ecotrack-env/bin/activate  # On Windows: ecotrack-env\Scripts\activate
   pip install -r requirements.txt
   ```

3. Download sample data (optional):
   ```bash
   python scripts/download_sample_data.py
   ```

## Usage

### Running in Demo Mode

The demo mode creates sample data and runs a full demonstration of EcoTrack's capabilities:

```bash
python main.py --mode demo --output-dir output
```

### Deforestation Analysis

Analyze deforestation between two satellite images:

```bash
python main.py --mode deforestation --before-image data/satellite/before_image.tif --after-image data/satellite/after_image.tif --output-dir output
```

### Pollution Analysis

Analyze pollution trends from sensor data:

```bash
python main.py --mode pollution --sensor-file data/sensor/air_quality_data.csv --pollutant-col pm25 --output-dir output
```

### Interactive Dashboard

Launch the interactive dashboard:

```bash
python main.py --mode dashboard --sensor-file data/sensor/air_quality_data.csv --dashboard-port 8050 --output-dir output
```

Then open a web browser and navigate to `http://localhost:8050` to view the dashboard.

## Command Line Arguments

- `--mode`: Operation mode (`demo`, `deforestation`, `pollution`, `dashboard`)
- `--satellite-dir`: Directory containing satellite imagery
- `--sensor-dir`: Directory containing sensor data
- `--output-dir`: Directory for output files
- `--before-image`: Path to "before" satellite image for deforestation analysis
- `--after-image`: Path to "after" satellite image for deforestation analysis
- `--sensor-file`: Path to sensor data file for pollution analysis
- `--pollutant-col`: Column name for pollutant in sensor data
- `--dashboard-port`: Port for the interactive dashboard

## Data Sources

### Satellite Imagery

EcoTrack supports various satellite imagery formats, including:
- Landsat (USGS)
- Sentinel (ESA Copernicus)
- MODIS (NASA)

You can download satellite imagery from:
- [USGS Earth Explorer](https://earthexplorer.usgs.gov/)
- [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
- [NASA Earthdata](https://earthdata.nasa.gov/)

### Sensor Data

The platform can work with various environmental sensor data in CSV or JSON format. The recommended data structure is:

```
timestamp,sensor_id,location,latitude,longitude,pm25,pm10,o3,no2,temperature,humidity
2022-01-01 00:00:00,sensor_1,Forest Edge,37.7749,-122.4194,8.32,16.45,22.1,12.3,15.2,68.5
...
```

## Core Modules

### Satellite Image Processing

The `SatelliteImageProcessor` class provides methods for:
- Loading satellite imagery
- Calculating vegetation indices (NDVI)
- Detecting deforestation
- Visualizing changes

### Deforestation Detection

The `DeforestationDetector` class implements multiple algorithms:
- NDVI difference analysis
- K-means classification
- Random Forest classification
- CNN-based detection

### Sensor Data Processing

The `SensorDataProcessor` class provides functionality for:
- Loading and cleaning sensor data
- Aggregating data by time periods
- Calculating pollution indices

### Pollution Analysis

The `PollutionAnalyzer` class offers methods for:
- Trend analysis
- Correlation analysis
- Time series forecasting (ARIMA/SARIMA)
- Visualization of pollution data

### Visualization

The `MapVisualizer` and `DataVisualizer` classes create various visualizations:
- Interactive maps with Folium
- Time series plots
- Heatmaps
- Bar charts and scatter plots
- Interactive dashboards with Dash

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Rasterio](https://github.com/mapbox/rasterio) for geospatial raster data processing
- [GeoPandas](https://github.com/geopandas/geopandas) for geospatial vector data processing
- [Folium](https://github.com/python-visualization/folium) for interactive maps
- [Dash](https://github.com/plotly/dash) for interactive dashboards
- [Scikit-learn](https://github.com/scikit-learn/scikit-learn) for machine learning algorithms
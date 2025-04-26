"""
EcoTrack - Visualization and Dashboard Components
This module contains functions for creating visualizations and interactive dashboards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import json
import os
from datetime import datetime, timedelta

class MapVisualizer:
    """Class for creating interactive maps with environmental data"""
    
    def __init__(self, base_map='OpenStreetMap'):
        """
        Initialize the map visualizer
        
        Parameters:
        -----------
        base_map : str
            Base map type for Folium maps
        """
        self.base_map = base_map
    
    def create_base_map(self, location=[0, 0], zoom_start=2, tiles=None):
        """
        Create a base map for visualization
        
        Parameters:
        -----------
        location : list
            [latitude, longitude] for initial map center
        zoom_start : int
            Initial zoom level
        tiles : str
            Map tile source
            
        Returns:
        --------
        folium.Map
            Base map
        """
        # Use default tiles if none specified
        if tiles is None:
            tiles = self.base_map
        
        # Create the map
        m = folium.Map(location=location, zoom_start=zoom_start, tiles=tiles)
        
        # Add scale bar
        folium.plugins.Scale().add_to(m)
        
        return m
    
    def add_point_data(self, m, data, lat_col='latitude', lon_col='longitude', 
                      popup_cols=None, tooltip=None, color_col=None, 
                      color_scheme='YlOrRd', radius=8, fill=True):
        """
        Add point data to the map
        
        Parameters:
        -----------
        m : folium.Map
            Base map
        data : pandas.DataFrame
            Data containing coordinates and attributes
        lat_col : str
            Column name for latitude
        lon_col : str
            Column name for longitude
        popup_cols : list
            List of column names to include in popups
        tooltip : str
            Column name to use for tooltip
        color_col : str
            Column name to use for coloring points
        color_scheme : str
            Colormap for points
        radius : int
            Size of circle markers
        fill : bool
            Whether to fill the markers
            
        Returns:
        --------
        folium.Map
            Map with point data added
        """
        # Create a copy of the data
        df = data.copy()
        
        # Set default popup columns if none specified
        if popup_cols is None:
            popup_cols = [col for col in df.columns if col not in [lat_col, lon_col]]
        
        # Check if we need a colormap
        if color_col is not None and color_col in df.columns:
            # Create colormap
            cmap = plt.cm.get_cmap(color_scheme)
            
            # Get min and max values for scaling
            vmin = df[color_col].min()
            vmax = df[color_col].max()
            
            # Create a colormap function
            def get_color(value):
                # Scale to 0-1
                scaled_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                # Get RGBA color and convert to hex
                rgba = cmap(scaled_value)
                r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
                return f'#{r:02x}{g:02x}{b:02x}'
            
            # Create a legend for the colormap
            legend_html = f'''
                <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
                    <h4>{color_col}</h4>
                    <div style="display: flex; flex-direction: row; align-items: center;">
                        <span>{vmin:.2f}</span>
                        <div style="width: 150px; height: 20px; margin: 0 10px; background: linear-gradient(to right, {get_color(vmin)}, {get_color(vmax)});"></div>
                        <span>{vmax:.2f}</span>
                    </div>
                </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add each point to the map
        for idx, row in df.iterrows():
            # Skip if lat/lon are missing
            if pd.isna(row[lat_col]) or pd.isna(row[lon_col]):
                continue
            
            # Create popup content
            if popup_cols:
                popup_content = "<table>"
                for col in popup_cols:
                    if col in row.index:
                        popup_content += f"<tr><td><b>{col}</b></td><td>{row[col]}</td></tr>"
                popup_content += "</table>"
                popup = folium.Popup(popup_content, max_width=300)
            else:
                popup = None
            
            # Create tooltip
            if tooltip and tooltip in row.index:
                tooltip_content = f"{tooltip}: {row[tooltip]}"
            else:
                tooltip_content = None
            
            # Determine color
            if color_col and color_col in row.index and not pd.isna(row[color_col]):
                color = get_color(row[color_col])
            else:
                color = 'blue'
            
            # Add marker
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=radius,
                popup=popup,
                tooltip=tooltip_content,
                color=color,
                fill=fill,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(m)
        
        return m
    
    def add_polygon_data(self, m, gdf, popup_cols=None, tooltip=None, color_col=None, 
                        color_scheme='YlOrRd', fill=True, stroke=True):
        """
        Add polygon data to the map
        
        Parameters:
        -----------
        m : folium.Map
            Base map
        gdf : geopandas.GeoDataFrame
            GeoDataFrame with polygon geometries
        popup_cols : list
            List of column names to include in popups
        tooltip : str
            Column name to use for tooltip
        color_col : str
            Column name to use for coloring polygons
        color_scheme : str
            Colormap for polygons
        fill : bool
            Whether to fill the polygons
        stroke : bool
            Whether to draw polygon borders
            
        Returns:
        --------
        folium.Map
            Map with polygon data added
        """
        # Set default popup columns if none specified
        if popup_cols is None:
            popup_cols = [col for col in gdf.columns if col != 'geometry']
        
        # Check if we need a colormap
        if color_col is not None and color_col in gdf.columns:
            # Create colormap
            cmap = plt.cm.get_cmap(color_scheme)
            
            # Get min and max values for scaling
            vmin = gdf[color_col].min()
            vmax = gdf[color_col].max()
            
            # Create a colormap function
            def get_color(value):
                # Scale to 0-1
                scaled_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                # Get RGBA color and convert to hex
                rgba = cmap(scaled_value)
                r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
                return f'#{r:02x}{g:02x}{b:02x}'
            
            # Create a legend for the colormap
            legend_html = f'''
                <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
                    <h4>{color_col}</h4>
                    <div style="display: flex; flex-direction: row; align-items: center;">
                        <span>{vmin:.2f}</span>
                        <div style="width: 150px; height: 20px; margin: 0 10px; background: linear-gradient(to right, {get_color(vmin)}, {get_color(vmax)});"></div>
                        <span>{vmax:.2f}</span>
                    </div>
                </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
        
        # Convert the GeoDataFrame to GeoJSON
        geojson_data = json.loads(gdf.to_json())
        
        # Define style function
        def style_function(feature):
            properties = feature['properties']
            
            # Get color
            if color_col and color_col in properties and properties[color_col] is not None:
                color = get_color(properties[color_col])
            else:
                color = 'blue'
            
            return {
                'fillColor': color,
                'color': 'black' if stroke else color,
                'weight': 1 if stroke else 0,
                'fillOpacity': 0.7 if fill else 0
            }
        
        # Define highlight function
        def highlight_function(feature):
            return {
                'weight': 3,
                'color': 'black',
                'fillOpacity': 0.9
            }
        
        # Define popup function
        def popup_function(feature):
            properties = feature['properties']
            popup_content = "<table>"
            for col in popup_cols:
                if col in properties and properties[col] is not None:
                    popup_content += f"<tr><td><b>{col}</b></td><td>{properties[col]}</td></tr>"
            popup_content += "</table>"
            return popup_content
        
        # Define tooltip function
        def tooltip_function(feature):
            properties = feature['properties']
            if tooltip and tooltip in properties and properties[tooltip] is not None:
                return f"{tooltip}: {properties[tooltip]}"
            return ''
        
        # Add GeoJSON to map
        folium.GeoJson(
            geojson_data,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(fields=[tooltip] if tooltip else [], labels=False),
            popup=folium.GeoJsonPopup(fields=popup_cols),
        ).add_to(m)
        
        return m
    
    def add_heatmap(self, m, data, lat_col='latitude', lon_col='longitude', 
                   weight_col=None, radius=15, blur=10, gradient=None,
                   min_opacity=0.2, max_zoom=18):
        """
        Add a heatmap layer to the map
        
        Parameters:
        -----------
        m : folium.Map
            Base map
        data : pandas.DataFrame
            Data containing coordinates and weight values
        lat_col : str
            Column name for latitude
        lon_col : str
            Column name for longitude
        weight_col : str
            Column name for heatmap intensity weights
        radius : int
            Radius of each point in the heatmap
        blur : int
            Amount of blur
        gradient : dict
            Custom color gradient
        min_opacity : float
            Minimum opacity of the heatmap
        max_zoom : int
            Maximum zoom level for the heatmap
            
        Returns:
        --------
        folium.Map
            Map with heatmap layer added
        """
        # Create a copy of the data
        df = data.copy()
        
        # Prepare data for heatmap
        if weight_col is not None and weight_col in df.columns:
            # Use specified weight column
            heat_data = [[row[lat_col], row[lon_col], row[weight_col]] 
                        for idx, row in df.iterrows() 
                        if not pd.isna(row[lat_col]) and not pd.isna(row[lon_col])]
        else:
            # No weights
            heat_data = [[row[lat_col], row[lon_col]] 
                        for idx, row in df.iterrows() 
                        if not pd.isna(row[lat_col]) and not pd.isna(row[lon_col])]
        
        # Set default gradient if none specified
        if gradient is None:
            gradient = {
                0.0: 'blue',
                0.2: 'green',
                0.4: 'yellow',
                0.6: 'orange',
                0.8: 'red',
                1.0: 'darkred'
            }
        
        # Add heatmap layer
        heatmap = plugins.HeatMap(
            heat_data,
            radius=radius,
            blur=blur,
            gradient=gradient,
            min_opacity=min_opacity,
            max_zoom=max_zoom
        )
        
        heatmap.add_to(m)
        
        return m
    
    def add_time_slider(self, m, gdf, time_col='timestamp', popup_cols=None, 
                       tooltip=None, color_col=None, color_scheme='YlOrRd'):
        """
        Add a time slider to animate data over time
        
        Parameters:
        -----------
        m : folium.Map
            Base map
        gdf : geopandas.GeoDataFrame
            GeoDataFrame with time-indexed data
        time_col : str
            Column name for timestamps
        popup_cols : list
            List of column names to include in popups
        tooltip : str
            Column name to use for tooltip
        color_col : str
            Column name to use for coloring
        color_scheme : str
            Colormap for points/polygons
            
        Returns:
        --------
        folium.Map
            Map with time slider
        """
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_dtype(gdf[time_col]):
            gdf = gdf.copy()
            gdf[time_col] = pd.to_datetime(gdf[time_col])
        
        # Set default popup columns if none specified
        if popup_cols is None:
            popup_cols = [col for col in gdf.columns if col not in ['geometry', time_col]]
        
        # Group data by time period
        grouped = {}
        for time_val, group in gdf.groupby(time_col):
            time_str = time_val.strftime('%Y-%m-%d')
            grouped[time_str] = group
        
        # Check if data contains points or polygons
        geom_type = gdf.geometry.iloc[0].geom_type
        is_point = geom_type in ['Point', 'MultiPoint']
        
        # Create feature groups for each time period
        feature_groups = {}
        for time_str, group in grouped.items():
            fg = folium.FeatureGroup(name=time_str)
            
            if is_point:
                # Add points
                for idx, row in group.iterrows():
                    # Create popup content
                    popup_content = "<table>"
                    for col in popup_cols:
                        if col in row.index:
                            popup_content += f"<tr><td><b>{col}</b></td><td>{row[col]}</td></tr>"
                    popup_content += "</table>"
                    popup = folium.Popup(popup_content, max_width=300)
                    
                    # Create tooltip
                    if tooltip and tooltip in row.index:
                        tooltip_content = f"{tooltip}: {row[tooltip]}"
                    else:
                        tooltip_content = None
                    
                    # Determine color
                    if color_col and color_col in row.index and not pd.isna(row[color_col]):
                        # Create colormap
                        cmap = plt.cm.get_cmap(color_scheme)
                        
                        # Get min and max values for scaling
                        vmin = gdf[color_col].min()
                        vmax = gdf[color_col].max()
                        
                        # Scale value to 0-1
                        scaled_value = (row[color_col] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                        
                        # Get RGBA color and convert to hex
                        rgba = cmap(scaled_value)
                        r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
                        color = f'#{r:02x}{g:02x}{b:02x}'
                    else:
                        color = 'blue'
                    
                    # Add marker
                    point = row.geometry.centroid
                    folium.CircleMarker(
                        location=[point.y, point.x],
                        radius=8,
                        popup=popup,
                        tooltip=tooltip_content,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7
                    ).add_to(fg)
            else:
                # Add polygons
                geojson_data = json.loads(group.to_json())
                
                # Define style function
                def style_function(feature):
                    properties = feature['properties']
                    
                    # Get color
                    if color_col and color_col in properties and properties[color_col] is not None:
                        # Create colormap
                        cmap = plt.cm.get_cmap(color_scheme)
                        
                        # Get min and max values for scaling
                        vmin = gdf[color_col].min()
                        vmax = gdf[color_col].max()
                        
                        # Scale value to 0-1
                        scaled_value = (properties[color_col] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                        
                        # Get RGBA color and convert to hex
                        rgba = cmap(scaled_value)
                        r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
                        color = f'#{r:02x}{g:02x}{b:02x}'
                    else:
                        color = 'blue'
                    
                    return {
                        'fillColor': color,
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    }
                
                # Add GeoJSON to feature group
                folium.GeoJson(
                    geojson_data,
                    style_function=style_function,
                    tooltip=folium.GeoJsonTooltip(fields=[tooltip] if tooltip else [], labels=False),
                    popup=folium.GeoJsonPopup(fields=popup_cols),
                ).add_to(fg)
            
            feature_groups[time_str] = fg
        
        # Add time slider
        time_index = sorted(feature_groups.keys())
        slider = plugins.TimestampedGeoJson(
            {
                "type": "FeatureCollection",
                "features": []
            },
            period="P1D",
            add_last_point=True,
            auto_play=True,
            transition_time=500,
            time_slider_drag_update=True
        )
        
        m.add_child(slider)
        
        # Add each feature group to the map
        for time_str, fg in feature_groups.items():
            m.add_child(fg)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def save_map(self, m, output_path):
        """
        Save the map as an HTML file
        
        Parameters:
        -----------
        m : folium.Map
            Map to save
        output_path : str
            Path to save the HTML file
        """
        m.save(output_path)
        print(f"Map saved to {output_path}")


class DataVisualizer:
    """Class for creating data visualizations"""
    
    def __init__(self, style='seaborn'):
        """
        Initialize the data visualizer
        
        Parameters:
        -----------
        style : str
            Matplotlib style to use
        """
        plt.style.use(style)
    
    def plot_time_series(self, data, y_col, x_col='timestamp', title=None, 
                        xlabel=None, ylabel=None, figsize=(12, 6), color='blue',
                        rolling_window=None, show_trend=False, output_path=None):
        """
        Create a time series plot
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data containing time series
        y_col : str or list
            Column name(s) for y-axis values
        x_col : str
            Column name for x-axis (time)
        title : str
            Plot title
        xlabel : str
            x-axis label
        ylabel : str
            y-axis label
        figsize : tuple
            Figure size
        color : str or list
            Line color(s)
        rolling_window : int
            Window size for rolling average (None for no smoothing)
        show_trend : bool
            Whether to show trend line
        output_path : str
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with time series plot
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Ensure time column is datetime
        df = data.copy()
        if not pd.api.types.is_datetime64_dtype(df[x_col]):
            df[x_col] = pd.to_datetime(df[x_col])
        
        # Convert y_col to list if it's a single string
        if isinstance(y_col, str):
            y_cols = [y_col]
        else:
            y_cols = y_col
        
        # Convert color to list if it's a single string and we have multiple y columns
        if isinstance(color, str) and len(y_cols) > 1:
            colors = [plt.cm.tab10(i) for i in range(len(y_cols))]
        elif isinstance(color, list):
            colors = color
        else:
            colors = [color] * len(y_cols)
        
        # Plot each y column
        for i, col in enumerate(y_cols):
            ax.plot(df[x_col], df[col], label=col, color=colors[i])
            
            # Add rolling average
            if rolling_window is not None and rolling_window > 1:
                rolling_avg = df[col].rolling(window=rolling_window).mean()
                ax.plot(df[x_col], rolling_avg, label=f'{col} ({rolling_window}-period MA)', 
                       linestyle='--', color=colors[i], alpha=0.7)
            
            # Add trend line
            if show_trend:
                # Create sequence for x values
                x_numeric = np.arange(len(df))
                
                # Fit trend line (linear regression)
                z = np.polyfit(x_numeric, df[col], 1)
                p = np.poly1d(z)
                trend = p(x_numeric)
                
                ax.plot(df[x_col], trend, linestyle='-.', 
                       color=colors[i], alpha=0.7, 
                       label=f'{col} trend (slope: {z[0]:.4f})')
        
        # Set title and labels
        if title:
            ax.set_title(title)
        
        ax.set_xlabel(xlabel if xlabel else x_col)
        ax.set_ylabel(ylabel if ylabel else ', '.join(y_cols))
        
        # Add legend if we have multiple lines
        if len(y_cols) > 1 or rolling_window is not None or show_trend:
            ax.legend()
        
        # Format x-axis for dates
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_boxplot(self, data, x_col=None, y_col=None, hue=None, title=None,
                   xlabel=None, ylabel=None, figsize=(10, 6), palette='Set3',
                   output_path=None):
        """
        Create a box plot
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data for the box plot
        x_col : str
            Column name for x-axis categories
        y_col : str
            Column name for y-axis values
        hue : str
            Column name for grouping
        title : str
            Plot title
        xlabel : str
            x-axis label
        ylabel : str
            y-axis label
        figsize : tuple
            Figure size
        palette : str
            Color palette
        output_path : str
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with box plot
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create box plot
        sns.boxplot(data=data, x=x_col, y=y_col, hue=hue, palette=palette, ax=ax)
        
        # Set title and labels
        if title:
            ax.set_title(title)
        
        if xlabel:
            ax.set_xlabel(xlabel)
        
        if ylabel:
            ax.set_ylabel(ylabel)
        
        # Format x-axis labels if needed
        if x_col and len(data[x_col].unique()) > 10:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_bar_chart(self, data, x_col, y_col, title=None, xlabel=None, ylabel=None,
                      figsize=(10, 6), color='skyblue', horizontal=False, 
                      sort_values=False, output_path=None):
        """
        Create a bar chart
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data for the bar chart
        x_col : str
            Column name for x-axis categories
        y_col : str
            Column name for y-axis values
        title : str
            Plot title
        xlabel : str
            x-axis label
        ylabel : str
            y-axis label
        figsize : tuple
            Figure size
        color : str
            Bar color
        horizontal : bool
            Whether to create a horizontal bar chart
        sort_values : bool
            Whether to sort bars by value
        output_path : str
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with bar chart
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a copy of the data
        df = data.copy()
        
        # Sort by value if requested
        if sort_values:
            df = df.sort_values(by=y_col)
        
        # Create bar chart
        if horizontal:
            ax.barh(df[x_col], df[y_col], color=color)
        else:
            ax.bar(df[x_col], df[y_col], color=color)
        
        # Set title and labels
        if title:
            ax.set_title(title)
        
        ax.set_xlabel(xlabel if xlabel else x_col)
        ax.set_ylabel(ylabel if ylabel else y_col)
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_heatmap(self, data, title=None, figsize=(10, 8), cmap='coolwarm',
                    annot=True, mask_upper=False, output_path=None):
        """
        Create a heatmap (often used for correlation matrices)
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data for the heatmap (e.g., correlation matrix)
        title : str
            Plot title
        figsize : tuple
            Figure size
        cmap : str
            Colormap
        annot : bool
            Whether to annotate cells
        mask_upper : bool
            Whether to mask the upper triangle
        output_path : str
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with heatmap
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create mask for upper triangle if requested
        if mask_upper:
            mask = np.triu(np.ones_like(data, dtype=bool))
        else:
            mask = None
        
        # Create heatmap
        sns.heatmap(data, mask=mask, cmap=cmap, annot=annot, fmt=".2f",
                   linewidths=0.5, ax=ax, vmin=-1 if data.values.min() < 0 else 0, vmax=1)
        
        # Set title
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_dashboard(self, port=8050, debug=False):
        """
        Create an interactive Dash dashboard
        
        Parameters:
        -----------
        port : int
            Port to run the dashboard on
        debug : bool
            Whether to run in debug mode
            
        Returns:
        --------
        dash.Dash
            Dash app
        """
        # Create Dash app
        app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
        
        # Define app layout (placeholder)
        app.layout = html.Div([
            html.H1("EcoTrack Environmental Dashboard", style={'textAlign': 'center'}),
            
            html.Div([
                html.Div([
                    html.Label("Select Data Source:"),
                    dcc.Dropdown(
                        id='data-source-dropdown',
                        options=[
                            {'label': 'Deforestation Data', 'value': 'deforestation'},
                            {'label': 'Air Quality Data', 'value': 'air_quality'},
                            {'label': 'Water Quality Data', 'value': 'water_quality'}
                        ],
                        value='deforestation'
                    )
                ], className='six columns'),
                
                html.Div([
                    html.Label("Select Time Range:"),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        start_date=datetime(2020, 1, 1),
                        end_date=datetime.now(),
                        max_date_allowed=datetime.now()
                    )
                ], className='six columns')
            ], className='row'),
            
            html.Div([
                html.Div([
                    html.Label("Select Visualization Type:"),
                    dcc.RadioItems(
                        id='viz-type-radio',
                        options=[
                            {'label': 'Time Series', 'value': 'time_series'},
                            {'label': 'Map View', 'value': 'map'},
                            {'label': 'Comparison', 'value': 'comparison'}
                        ],
                        value='time_series',
                        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                    )
                ], className='twelve columns')
            ], className='row'),
            
            html.Div([
                dcc.Loading(
                    id="loading-1",
                    type="default",
                    children=html.Div(id='visualization-container')
                )
            ], className='row', style={'marginTop': 20}),
            
            html.Div([
                html.Div([
                    html.H3("Statistics", style={'textAlign': 'center'}),
                    html.Div(id='stats-container')
                ], className='twelve columns')
            ], className='row', style={'marginTop': 20})
        ])
        
        # Define callback to update visualization
        @app.callback(
            Output('visualization-container', 'children'),
            [
                Input('data-source-dropdown', 'value'),
                Input('date-picker-range', 'start_date'),
                Input('date-picker-range', 'end_date'),
                Input('viz-type-radio', 'value')
            ]
        )
        def update_visualization(data_source, start_date, end_date, viz_type):
            # This is a placeholder for the actual implementation
            
            if viz_type == 'time_series':
                # Example time series figure
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pd.date_range(start=start_date, end=end_date, freq='M'),
                    y=np.random.randn(len(pd.date_range(start=start_date, end=end_date, freq='M'))).cumsum(),
                    mode='lines+markers',
                    name=f'{data_source} Trend'
                ))
                fig.update_layout(
                    title=f'{data_source.title()} Time Series',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    template='plotly_white'
                )
                return dcc.Graph(figure=fig)
            
            elif viz_type == 'map':
                # Example map figure (placeholder)
                fig = go.Figure(go.Scattermapbox(
                    lat=[0],
                    lon=[0],
                    mode='markers',
                    marker=go.scattermapbox.Marker(size=14),
                    text=['Example Location']
                ))
                
                fig.update_layout(
                    title=f'{data_source.title()} Map View',
                    mapbox_style="open-street-map",
                    mapbox=dict(
                        center=go.layout.mapbox.Center(lat=0, lon=0),
                        zoom=1
                    ),
                    margin={"r": 0, "t": 40, "l": 0, "b": 0}
                )
                return dcc.Graph(figure=fig)
            
            elif viz_type == 'comparison':
                # Example comparison figure
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Before", "After"))
                
                fig.add_trace(
                    go.Bar(x=['Area 1', 'Area 2', 'Area 3'], y=[10, 15, 13], name="Before"),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=['Area 1', 'Area 2', 'Area 3'], y=[12, 9, 17], name="After"),
                    row=1, col=2
                )
                
                fig.update_layout(
                    title_text=f'{data_source.title()} Comparison',
                    showlegend=False
                )
                return dcc.Graph(figure=fig)
            
            return html.Div("Please select a visualization type")
        
        # Define callback to update statistics
        @app.callback(
            Output('stats-container', 'children'),
            [
                Input('data-source-dropdown', 'value'),
                Input('date-picker-range', 'start_date'),
                Input('date-picker-range', 'end_date')
            ]
        )
        def update_stats(data_source, start_date, end_date):
            # This is a placeholder for the actual implementation
            
            if data_source == 'deforestation':
                stats = [
                    {'name': 'Total Deforested Area', 'value': '1,234 km²'},
                    {'name': 'Deforestation Rate', 'value': '2.3% per year'},
                    {'name': 'Most Affected Region', 'value': 'Region A'},
                    {'name': 'Risk Level', 'value': 'High'}
                ]
            elif data_source == 'air_quality':
                stats = [
                    {'name': 'Average AQI', 'value': '85'},
                    {'name': 'Days Above Threshold', 'value': '12'},
                    {'name': 'Main Pollutant', 'value': 'PM2.5'},
                    {'name': 'Trend', 'value': 'Improving'}
                ]
            else:  # water_quality
                stats = [
                    {'name': 'Average pH', 'value': '7.2'},
                    {'name': 'Contamination Incidents', 'value': '3'},
                    {'name': 'Water Quality Index', 'value': '82/100'},
                    {'name': 'Trend', 'value': 'Stable'}
                ]
            
            # Create stats cards
            return html.Div([
                html.Div([
                    html.Div([
                        html.H4(stat['name']),
                        html.P(stat['value'], style={'fontSize': '24px', 'fontWeight': 'bold'})
                    ], className='card', style={
                        'border': '1px solid #ddd',
                        'borderRadius': '5px',
                        'padding': '10px',
                        'margin': '10px',
                        'textAlign': 'center',
                        'boxShadow': '2px 2px 2px lightgrey'
                    })
                ], className='three columns') for stat in stats
            ], className='row')
        
        # Return the app
        return app
    
    def run_dashboard(self, app, port=8050, debug=False):
        """
        Run the Dash dashboard
        
        Parameters:
        -----------
        app : dash.Dash
            Dash app to run
        port : int
            Port to run the dashboard on
        debug : bool
            Whether to run in debug mode
        """
        app.run(port=port, debug=debug, host='0.0.0.0')


class PlotlyVisualizer:
    """Class for creating interactive Plotly visualizations"""
    
    def create_time_series(self, data, y_col, x_col='timestamp', title=None, 
                          xlabel=None, ylabel=None, color_col=None, 
                          add_trend=False, height=500):
        """
        Create an interactive time series plot with Plotly
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data containing time series
        y_col : str or list
            Column name(s) for y-axis values
        x_col : str
            Column name for x-axis (time)
        title : str
            Plot title
        xlabel : str
            x-axis label
        ylabel : str
            y-axis label
        color_col : str
            Column to use for color grouping
        add_trend : bool
            Whether to add trend lines
        height : int
            Plot height
            
        Returns:
        --------
        plotly.graph_objs._figure.Figure
            Plotly figure
        """
        # Convert y_col to list if it's a single string
        if isinstance(y_col, str):
            y_cols = [y_col]
        else:
            y_cols = y_col
        
        # Ensure time column is datetime
        df = data.copy()
        if not pd.api.types.is_datetime64_dtype(df[x_col]):
            df[x_col] = pd.to_datetime(df[x_col])
        
        # Create figure
        if color_col and color_col in df.columns:
            # Create grouped time series
            fig = px.line(
                df, x=x_col, y=y_cols, color=color_col,
                title=title,
                labels={col: col.replace('_', ' ').title() for col in df.columns},
                height=height
            )
        else:
            # Create simple time series
            fig = go.Figure()
            
            for col in y_cols:
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[col],
                    mode='lines+markers',
                    name=col.replace('_', ' ').title()
                ))
            
            # Add trend lines if requested
            if add_trend:
                for col in y_cols:
                    # Create sequence for x values
                    x_numeric = np.arange(len(df))
                    
                    # Fit trend line (linear regression)
                    z = np.polyfit(x_numeric, df[col], 1)
                    p = np.poly1d(z)
                    trend = p(x_numeric)
                    
                    fig.add_trace(go.Scatter(
                        x=df[x_col],
                        y=trend,
                        mode='lines',
                        line=dict(dash='dash'),
                        name=f'{col} trend',
                        showlegend=True
                    ))
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=xlabel if xlabel else x_col.replace('_', ' ').title(),
                yaxis_title=ylabel if ylabel else ', '.join([col.replace('_', ' ').title() for col in y_cols]),
                legend_title="Variable",
                hovermode="x unified",
                height=height
            )
        
        return fig
    
    def create_choropleth_map(self, gdf, value_col, geo_id_col, title=None,
                             colorscale='Viridis', hover_data=None, height=600):
        """
        Create an interactive choropleth map with Plotly
        
        Parameters:
        -----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame with geometries
        value_col : str
            Column name for choropleth values
        geo_id_col : str
            Column name for geometry IDs
        title : str
            Plot title
        colorscale : str
            Colorscale name
        hover_data : list
            List of columns to show in hover data
        height : int
            Plot height
            
        Returns:
        --------
        plotly.graph_objs._figure.Figure
            Plotly figure
        """
        # Convert GeoDataFrame to GeoJSON
        geojson_data = json.loads(gdf.to_json())
        
        # Create choropleth map
        fig = px.choropleth_mapbox(
            gdf,
            geojson=geojson_data,
            locations=gdf.index,
            color=value_col,
            color_continuous_scale=colorscale,
            hover_data=hover_data,
            mapbox_style="carto-positron",
            zoom=3,
            center={"lat": gdf.geometry.centroid.y.mean(), "lon": gdf.geometry.centroid.x.mean()},
            opacity=0.5,
            height=height
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            margin={"r": 0, "t": 30, "l": 0, "b": 0}
        )
        
        return fig
    
    def create_3d_surface(self, df, x_col, y_col, z_col, title=None, height=700):
        """
        Create a 3D surface plot with Plotly
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Data for the surface plot
        x_col : str
            Column name for x-axis values
        y_col : str
            Column name for y-axis values
        z_col : str
            Column name for z-axis values (surface height)
        title : str
            Plot title
        height : int
            Plot height
            
        Returns:
        --------
        plotly.graph_objs._figure.Figure
            Plotly figure
        """
        # Pivot data for surface plot
        try:
            # Try to pivot the data
            pivot_table = df.pivot(index=y_col, columns=x_col, values=z_col)
            
            # Create 3D surface plot
            fig = go.Figure(data=[go.Surface(z=pivot_table.values)])
            
            # Update layout
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title=x_col.replace('_', ' ').title(),
                    yaxis_title=y_col.replace('_', ' ').title(),
                    zaxis_title=z_col.replace('_', ' ').title()
                ),
                height=height
            )
            
            return fig
            
        except ValueError:
            # If pivoting fails, create a 3D scatter plot instead
            print("Could not create surface plot. Creating 3D scatter plot instead.")
            
            fig = px.scatter_3d(
                df, x=x_col, y=y_col, z=z_col,
                color=z_col,
                title=title,
                height=height
            )
            
            return fig
            
    def plot_scatter(self, data, x_col, y_col, color_col=None, size_col=None,
                     title=None, xlabel=None, ylabel=None, figsize=(10, 6),
                     alpha=0.7, add_trendline=False, output_path=None):
        """
        Create a scatter plot
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data for the scatter plot
        x_col : str
            Column name for x-axis values
        y_col : str
            Column name for y-axis values
        color_col : str
            Column name for point colors
        size_col : str
            Column name for point sizes
        title : str
            Plot title
        xlabel : str
            x-axis label
        ylabel : str
            y-axis label
        figsize : tuple
            Figure size
        alpha : float
            Point transparency
        add_trendline : bool
            Whether to add a trend line
        output_path : str
            Path to save the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with scatter plot
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine point colors
        if color_col and color_col in data.columns:
            colors = data[color_col]
            scatter = ax.scatter(data[x_col], data[y_col], 
                               c=colors, cmap='viridis', 
                               alpha=alpha,
                               s=data[size_col] if size_col and size_col in data.columns else 50)
            
            # Add color bar
            cbar = plt.colorbar(scatter)
            cbar.set_label(color_col)
        else:
            ax.scatter(data[x_col], data[y_col], 
                     alpha=alpha, 
                     s=data[size_col] if size_col and size_col in data.columns else 50)
        
        # Add trend line
        if add_trendline:
            # Fit trend line (linear regression)
            z = np.polyfit(data[x_col], data[y_col], 1)
            p = np.poly1d(z)
            
            # Add line to plot
            x_line = np.linspace(data[x_col].min(), data[x_col].max(), 100)
            y_line = p(x_line)
            ax.plot(x_line, y_line, 'r--', alpha=0.7, 
                   label=f'Trend: y = {z[0]:.4f}x + {z[1]:.4f}')
            ax.legend()
        
        # Set title and labels
        if title:
            ax.set_title(title)
        
        ax.set_xlabel(xlabel if xlabel else x_col)
        ax.set_ylabel(ylabel if ylabel else y_col)
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
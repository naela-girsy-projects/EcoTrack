"""
EcoTrack - Pollution Monitoring and Trend Analysis
This module contains functions for analyzing pollution data and forecasting trends.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import joblib
import os
from datetime import datetime, timedelta

class PollutionAnalyzer:
    """Class for analyzing and forecasting pollution data"""
    
    def __init__(self, model_dir=None):
        """
        Initialize the pollution analyzer
        
        Parameters:
        -----------
        model_dir : str
            Directory to save/load trained models
        """
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        
        # Create model directory if it doesn't exist
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def preprocess_data(self, data, target_column, time_column='timestamp', categorical_columns=None, drop_columns=None):
        """
        Preprocess pollution data for analysis
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw pollution data
        target_column : str
            Column name of the target variable
        time_column : str
            Column containing timestamp data
        categorical_columns : list
            List of categorical column names for encoding
        drop_columns : list
            List of columns to drop
            
        Returns:
        --------
        tuple
            (X, y, preprocessed_data)
        """
        # Make a copy of the data
        df = data.copy()
        
        # Ensure time column is datetime
        if time_column in df.columns and not pd.api.types.is_datetime64_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column])
        
        # Extract temporal features if time column exists
        if time_column in df.columns:
            df['hour'] = df[time_column].dt.hour
            df['day_of_week'] = df[time_column].dt.dayofweek
            df['month'] = df[time_column].dt.month
            df['day'] = df[time_column].dt.day
            df['year'] = df[time_column].dt.year
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
            
            # Create lags for the target variable
            for lag in [1, 3, 6, 12, 24]:
                lag_col = f'{target_column}_lag_{lag}'
                df[lag_col] = df[target_column].shift(lag)
            
            # Create rolling statistics
            for window in [3, 6, 12, 24]:
                # Rolling mean
                mean_col = f'{target_column}_rolling_mean_{window}'
                df[mean_col] = df[target_column].rolling(window=window).mean()
                
                # Rolling std
                std_col = f'{target_column}_rolling_std_{window}'
                df[std_col] = df[target_column].rolling(window=window).std()
        
        # Handle categorical columns
        if categorical_columns:
            for col in categorical_columns:
                if col in df.columns:
                    # One-hot encode
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    df.drop(columns=[col], inplace=True)
        
        # Drop specified columns
        if drop_columns:
            df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)
        
        # Drop rows with missing values
        df.dropna(inplace=True)
        
        # Split into features and target
        y = df[target_column] if target_column in df.columns else None
        
        # Drop target from features
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
        else:
            X = df
        
        return X, y, df
    
    def train_xgboost_model(self, X, y, test_size=0.2, random_state=42, params=None):
        """
        Train an XGBoost model for pollution prediction
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature data
        y : pandas.Series
            Target variable
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
        params : dict
            XGBoost parameters
            
        Returns:
        --------
        tuple
            (trained_model, X_train, X_test, y_train, y_test, feature_importance)
        """
        # Default XGBoost parameters
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.01,
                'reg_lambda': 1,
                'random_state': random_state
            }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Only scale numerical features
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
        
        # Save scaler
        self.scalers['xgboost'] = scaler
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], early_stopping_rounds=10, verbose=False)
        
        # Get feature importance
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Save model
        if self.model_dir:
            model_path = os.path.join(self.model_dir, 'xgboost_model.pkl')
            scaler_path = os.path.join(self.model_dir, 'xgboost_scaler.pkl')
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            print(f"Model saved to {model_path}")
        
        self.models['xgboost'] = model
        
        return model, X_train_scaled, X_test_scaled, y_train, y_test, feature_importance
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        model : trained model
            Model to evaluate
        X_test : pandas.DataFrame
            Test features
        y_test : pandas.Series
            Test target
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Create a DataFrame with actual and predicted values
        eval_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        
        # Calculate additional metrics
        eval_df['Error'] = eval_df['Actual'] - eval_df['Predicted']
        eval_df['Percent_Error'] = (eval_df['Error'] / eval_df['Actual']) * 100
        
        mape = np.mean(np.abs(eval_df['Percent_Error']))
        
        # Return metrics
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        return metrics, eval_df
    
    def predict_pollution(self, X, model_type='xgboost', model=None):
        """
        Make pollution predictions using a trained model
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature data
        model_type : str
            Type of model to use
        model : trained model
            Model to use (if None, uses stored model)
            
        Returns:
        --------
        numpy.ndarray
            Predicted values
        """
        # Use stored model if none provided
        if model is None:
            if model_type in self.models:
                model = self.models[model_type]
            else:
                print(f"No {model_type} model available. Please train or provide a model.")
                return None
        
        # Scale features if needed
        if model_type in self.scalers:
            scaler = self.scalers[model_type]
            X_scaled = X.copy()
            
            # Only scale numerical features
            num_cols = X.select_dtypes(include=[np.number]).columns
            X_scaled[num_cols] = scaler.transform(X[num_cols])
        else:
            X_scaled = X
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        return predictions
    
    def train_arima_model(self, data, target_column, time_column='timestamp', order=(1,1,1), seasonal_order=(1,1,1,12)):
        """
        Train an ARIMA/SARIMA model for time series forecasting
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Time series data
        target_column : str
            Column name of the target variable
        time_column : str
            Column containing timestamp data
        order : tuple
            (p, d, q) order for ARIMA
        seasonal_order : tuple
            (P, D, Q, s) seasonal order for SARIMA
            
        Returns:
        --------
        statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper
            Trained ARIMA/SARIMA model
        """
        # Ensure data is sorted by time
        if time_column in data.columns:
            df = data.sort_values(time_column).copy()
            df.set_index(time_column, inplace=True)
        else:
            df = data.copy()
        
        # Check if data has seasonality
        has_seasonality = seasonal_order[0] > 0 or seasonal_order[1] > 0 or seasonal_order[2] > 0
        
        # Train model
        if has_seasonality:
            # Train SARIMA
            model = SARIMAX(
                df[target_column],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=False)
            model_type = 'sarima'
        else:
            # Train ARIMA
            model = ARIMA(df[target_column], order=order)
            results = model.fit()
            model_type = 'arima'
        
        # Save model
        if self.model_dir:
            model_path = os.path.join(self.model_dir, f'{model_type}_model.pkl')
            joblib.dump(results, model_path)
            print(f"Model saved to {model_path}")
        
        self.models[model_type] = results
        
        return results
    
    def forecast_with_arima(self, model, steps=30, confidence_interval=0.95, model_type='sarima'):
        """
        Generate forecasts using an ARIMA/SARIMA model
        
        Parameters:
        -----------
        model : statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper
            Trained ARIMA/SARIMA model
        steps : int
            Number of steps to forecast
        confidence_interval : float
            Confidence interval level
        model_type : str
            Type of model ('arima' or 'sarima')
            
        Returns:
        --------
        pandas.DataFrame
            Forecasted values with confidence intervals
        """
        # Use stored model if none provided
        if model is None:
            if model_type in self.models:
                model = self.models[model_type]
            else:
                print(f"No {model_type} model available. Please train or provide a model.")
                return None
        
        # Generate forecasts
        forecast = model.get_forecast(steps=steps)
        
        # Get forecast values and confidence intervals
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=1-confidence_interval)
        
        # Create DataFrame with forecasts
        forecast_df = pd.DataFrame({
            'forecast': forecast_values,
            'lower_ci': conf_int.iloc[:, 0],
            'upper_ci': conf_int.iloc[:, 1]
        })
        
        return forecast_df
    
    def seasonal_decomposition(self, data, target_column, time_column='timestamp', period=12, model='additive'):
        """
        Perform seasonal decomposition of time series data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Time series data
        target_column : str
            Column name of the target variable
        time_column : str
            Column containing timestamp data
        period : int
            Period of seasonality
        model : str
            Type of seasonal model ('additive' or 'multiplicative')
            
        Returns:
        --------
        statsmodels.tsa.seasonal.DecomposeResult
            Decomposition results
        """
        # Ensure data is sorted by time
        if time_column in data.columns:
            df = data.sort_values(time_column).copy()
            df.set_index(time_column, inplace=True)
        else:
            df = data.copy()
        
        # Ensure data has no missing values
        df = df.dropna(subset=[target_column])
        
        # Perform decomposition
        decomposition = seasonal_decompose(df[target_column], model=model, period=period)
        
        return decomposition
    
    def visualize_forecast(self, historical_data, forecast_data, target_column, time_column='timestamp',
                          title='Pollution Forecast', figsize=(12, 6)):
        """
        Visualize historical data and forecasts
        
        Parameters:
        -----------
        historical_data : pandas.DataFrame
            Historical time series data
        forecast_data : pandas.DataFrame
            Forecasted data from ARIMA/SARIMA model
        target_column : str
            Column name of the target variable
        time_column : str
            Column containing timestamp data
        title : str
            Plot title
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Plot figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot historical data
        if time_column in historical_data.columns:
            hist_data = historical_data.sort_values(time_column).copy()
            hist_data.set_index(time_column, inplace=True)
        else:
            hist_data = historical_data.copy()
        
        ax.plot(hist_data.index, hist_data[target_column], label='Historical Data', color='blue')
        
        # Get last date from historical data
        last_date = hist_data.index[-1]
        
        # Create date range for forecast
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast_data))
        
        # Plot forecast
        ax.plot(forecast_dates, forecast_data['forecast'], label='Forecast', color='red')
        
        # Plot confidence intervals
        ax.fill_between(
            forecast_dates,
            forecast_data['lower_ci'],
            forecast_data['upper_ci'],
            color='red',
            alpha=0.2,
            label='Confidence Interval'
        )
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel(target_column)
        ax.set_title(title)
        ax.legend()
        
        # Format x-axis
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def analyze_pollution_trends(self, data, pollutants, time_column='timestamp', freq='M', agg_func='mean'):
        """
        Analyze trends in pollution data over time
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Pollution data
        pollutants : list
            List of pollutant column names
        time_column : str
            Column containing timestamp data
        freq : str
            Frequency for resampling (e.g., 'D' for daily, 'M' for monthly)
        agg_func : str or dict
            Aggregation function(s) for resampling
            
        Returns:
        --------
        pandas.DataFrame
            Resampled data with trends
        """
        # Ensure data is sorted by time
        if time_column in data.columns:
            df = data.sort_values(time_column).copy()
            df.set_index(time_column, inplace=True)
        else:
            df = data.copy()
        
        # Select only the pollutant columns
        pollutant_data = df[pollutants]
        
        # Resample data
        if isinstance(agg_func, str):
            resampled = pollutant_data.resample(freq).agg(agg_func)
        else:
            resampled = pollutant_data.resample(freq).agg(agg_func)
        
        # Calculate rolling statistics
        for pollutant in pollutants:
            # Calculate moving average
            resampled[f'{pollutant}_MA_3'] = resampled[pollutant].rolling(window=3).mean()
            resampled[f'{pollutant}_MA_6'] = resampled[pollutant].rolling(window=6).mean()
            
            # Calculate trend (using simple linear regression)
            X = np.arange(len(resampled)).reshape(-1, 1)
            y = resampled[pollutant].values
            
            # Handle missing values
            mask = ~np.isnan(y)
            if np.sum(mask) > 1:  # Need at least 2 points for regression
                X_valid = X[mask]
                y_valid = y[mask]
                
                # Fit trend line
                model = sm.OLS(y_valid, sm.add_constant(X_valid)).fit()
                
                # Store trend coefficient
                trend_coef = model.params[1]
                resampled[f'{pollutant}_trend'] = trend_coef
                
                # Calculate trend line values
                trend_line = model.params[0] + model.params[1] * X
                resampled[f'{pollutant}_trend_line'] = trend_line
        
        return resampled
    
    def correlation_analysis(self, data, columns, time_column=None, method='pearson'):
        """
        Analyze correlations between variables
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data for analysis
        columns : list
            List of column names to include in correlation analysis
        time_column : str
            Column containing timestamp data (to be excluded)
        method : str
            Correlation method ('pearson', 'spearman', or 'kendall')
            
        Returns:
        --------
        pandas.DataFrame
            Correlation matrix
        """
        # Create a copy of the data with selected columns
        selected_cols = [col for col in columns if col in data.columns and col != time_column]
        df = data[selected_cols].copy()
        
        # Calculate correlation matrix
        corr_matrix = df.corr(method=method)
        
        return corr_matrix
    
    def visualize_correlations(self, corr_matrix, figsize=(10, 8), cmap='coolwarm', annot=True):
        """
        Visualize correlation matrix as a heatmap
        
        Parameters:
        -----------
        corr_matrix : pandas.DataFrame
            Correlation matrix
        figsize : tuple
            Figure size
        cmap : str
            Colormap for heatmap
        annot : bool
            Whether to annotate cells with correlation values
            
        Returns:
        --------
        matplotlib.figure.Figure
            Heatmap figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        heatmap = sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmin=-1, vmax=1,
            annot=annot,
            fmt=".2f",
            linewidths=0.5,
            ax=ax
        )
        
        # Set title
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        return fig
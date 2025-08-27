import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import logging
from typing import Dict, List, Any, Optional, Tuple
import io
import warnings
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TimeSeriesForecaster:
    """
    Time-series forecasting module with Prophet and ARIMA models
    """
    
    def __init__(self):
        """Initialize the forecaster"""
        self.prophet_model = None
        self.arima_model = None
        self.data = None
        self.forecast_data = None
        self.anomalies = []
        
    def forecast(self, csv_content: bytes, filename: str, target_column: str, 
                forecast_periods: int = 30, detect_anomalies: bool = True) -> Dict[str, Any]:
        """
        Main forecasting function
        
        Args:
            csv_content: Raw CSV bytes
            filename: Original filename
            target_column: Column to forecast
            forecast_periods: Number of periods to forecast
            detect_anomalies: Whether to detect anomalies
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Load and preprocess data
            self.data = self._load_csv_data(csv_content, target_column)
            
            if self.data is None or len(self.data) == 0:
                return {
                    "success": False,
                    "error": "Invalid or empty data",
                    "filename": filename
                }
            
            # Validate data
            validation_result = self._validate_data()
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "filename": filename
                }
            
            # Generate forecasts
            prophet_forecast = self._prophet_forecast(forecast_periods)
            arima_forecast = self._arima_forecast(forecast_periods)
            
            # Combine forecasts
            combined_forecast = self._combine_forecasts(prophet_forecast, arima_forecast)
            
            # Detect anomalies if requested
            if detect_anomalies:
                self.anomalies = self._detect_anomalies(combined_forecast)
            
            # Prepare results
            results = {
                "success": True,
                "filename": filename,
                "target_column": target_column,
                "data_points": len(self.data),
                "forecast_periods": forecast_periods,
                "forecast": combined_forecast,
                "anomalies": self.anomalies,
                "model_performance": self._evaluate_models(),
                "data_summary": self._get_data_summary()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in forecasting: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "filename": filename
            }
    
    def _load_csv_data(self, csv_content: bytes, target_column: str) -> Optional[pd.DataFrame]:
        """Load CSV data and extract time series"""
        try:
            # Read CSV from bytes
            df = pd.read_csv(io.BytesIO(csv_content))
            
            if target_column not in df.columns:
                logger.error(f"Target column '{target_column}' not found in CSV")
                return None
            
            # Try to identify datetime column
            datetime_column = self._find_datetime_column(df)
            
            if datetime_column:
                # Convert to datetime and set as index
                df[datetime_column] = pd.to_datetime(df[datetime_column], errors='coerce')
                df = df.dropna(subset=[datetime_column])
                df = df.set_index(datetime_column)
                
                # Sort by datetime
                df = df.sort_index()
                
                # Select target column and create time series
                ts_data = df[[target_column]].copy()
                ts_data.columns = ['value']
                
                return ts_data
            else:
                # If no datetime column, create one based on row index
                ts_data = df[[target_column]].copy()
                ts_data.columns = ['value']
                ts_data.index = pd.date_range(
                    start='2020-01-01',
                    periods=len(ts_data),
                    freq='D'
                )
                return ts_data
                
        except Exception as e:
            logger.error(f"Error loading CSV data: {str(e)}")
            return None
    
    def _find_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find datetime column in dataframe"""
        datetime_columns = []
        
        for col in df.columns:
            # Check if column name suggests datetime
            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp', 'datetime']):
                datetime_columns.append(col)
            else:
                # Try to convert first few values to datetime
                try:
                    pd.to_datetime(df[col].head(), errors='raise')
                    datetime_columns.append(col)
                except:
                    continue
        
        return datetime_columns[0] if datetime_columns else None
    
    def _validate_data(self) -> Dict[str, Any]:
        """Validate time series data"""
        try:
            if self.data is None or len(self.data) < 10:
                return {"valid": False, "error": "Insufficient data points (minimum 10 required)"}
            
            # Check for missing values
            missing_count = self.data['value'].isnull().sum()
            if missing_count > 0:
                # Fill missing values with forward fill
                self.data = self.data.fillna(method='ffill')
                logger.info(f"Filled {missing_count} missing values")
            
            # Check for constant data
            if self.data['value'].std() == 0:
                return {"valid": False, "error": "Data is constant (no variation)"}
            
            # Check for stationarity
            stationarity_test = self._check_stationarity()
            if not stationarity_test["stationary"]:
                logger.warning("Data may not be stationary")
            
            return {"valid": True, "stationary": stationarity_test["stationary"]}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def _check_stationarity(self) -> Dict[str, Any]:
        """Check if time series is stationary using Augmented Dickey-Fuller test"""
        try:
            # Perform ADF test
            result = adfuller(self.data['value'].dropna())
            
            # Extract test statistic and p-value
            adf_statistic = result[0]
            p_value = result[1]
            
            # Critical values
            critical_values = result[4]
            
            is_stationary = p_value < 0.05
            
            return {
                "stationary": is_stationary,
                "adf_statistic": adf_statistic,
                "p_value": p_value,
                "critical_values": critical_values
            }
            
        except Exception as e:
            logger.warning(f"Could not perform stationarity test: {str(e)}")
            return {"stationary": True, "error": str(e)}
    
    def _prophet_forecast(self, periods: int) -> Dict[str, Any]:
        """Generate forecast using Facebook Prophet"""
        try:
            # Prepare data for Prophet
            prophet_df = self.data.reset_index()
            prophet_df.columns = ['ds', 'y']
            
            # Initialize and fit Prophet model
            self.prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            
            self.prophet_model.fit(prophet_df)
            
            # Generate future dates
            future = self.prophet_model.make_future_dataframe(periods=periods)
            
            # Make prediction
            forecast = self.prophet_model.predict(future)
            
            # Extract forecast values
            forecast_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            
            return {
                "method": "Prophet",
                "forecast": forecast_values.to_dict('records'),
                "model": self.prophet_model
            }
            
        except Exception as e:
            logger.error(f"Error in Prophet forecast: {str(e)}")
            return {"method": "Prophet", "error": str(e)}
    
    def _arima_forecast(self, periods: int) -> Dict[str, Any]:
        """Generate forecast using ARIMA model"""
        try:
            # Determine ARIMA parameters using auto_arima approach
            p, d, q = self._auto_arima_params()
            
            # Fit ARIMA model
            self.arima_model = ARIMA(self.data['value'], order=(p, d, q))
            fitted_model = self.arima_model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=periods)
            
            # Create forecast dataframe
            last_date = self.data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=periods,
                freq='D'
            )
            
            forecast_df = pd.DataFrame({
                'ds': forecast_dates,
                'yhat': forecast.values,
                'yhat_lower': forecast.values * 0.9,  # Simple confidence interval
                'yhat_upper': forecast.values * 1.1
            })
            
            return {
                "method": "ARIMA",
                "forecast": forecast_df.to_dict('records'),
                "model": fitted_model,
                "params": (p, d, q)
            }
            
        except Exception as e:
            logger.error(f"Error in ARIMA forecast: {str(e)}")
            return {"method": "ARIMA", "error": str(e)}
    
    def _auto_arima_params(self) -> Tuple[int, int, int]:
        """Automatically determine ARIMA parameters"""
        try:
            # Simple approach: try common parameter combinations
            best_aic = float('inf')
            best_params = (1, 1, 1)
            
            # Test different parameter combinations
            for p in [0, 1, 2]:
                for d in [0, 1, 2]:
                    for q in [0, 1, 2]:
                        try:
                            model = ARIMA(self.data['value'], order=(p, d, q))
                            fitted = model.fit()
                            aic = fitted.aic
                            
                            if aic < best_aic:
                                best_aic = aic
                                best_params = (p, d, q)
                        except:
                            continue
            
            return best_params
            
        except Exception as e:
            logger.warning(f"Error in auto ARIMA: {str(e)}, using default (1,1,1)")
            return (1, 1, 1)
    
    def _combine_forecasts(self, prophet_result: Dict, arima_result: Dict) -> List[Dict[str, Any]]:
        """Combine forecasts from both models"""
        combined = []
        
        # Get the number of periods from either model
        periods = 0
        if "forecast" in prophet_result and not "error" in prophet_result:
            periods = len(prophet_result["forecast"])
        elif "forecast" in arima_result and not "error" in arima_result:
            periods = len(arima_result["forecast"])
        
        for i in range(periods):
            combined_forecast = {
                "period": i + 1,
                "prophet": prophet_result.get("forecast", [{}])[i] if "forecast" in prophet_result else {},
                "arima": arima_result.get("forecast", [{}])[i] if "forecast" in arima_result else {},
                "ensemble": {}
            }
            
            # Calculate ensemble forecast (simple average)
            prophet_val = prophet_result.get("forecast", [{}])[i].get("yhat", None) if "forecast" in prophet_result else None
            arima_val = arima_result.get("forecast", [{}])[i].get("yhat", None) if "forecast" in arima_result else None
            
            if prophet_val is not None and arima_val is not None:
                combined_forecast["ensemble"]["yhat"] = (prophet_val + arima_val) / 2
                combined_forecast["ensemble"]["method"] = "Average"
            elif prophet_val is not None:
                combined_forecast["ensemble"]["yhat"] = prophet_val
                combined_forecast["ensemble"]["method"] = "Prophet only"
            elif arima_val is not None:
                combined_forecast["ensemble"]["yhat"] = arima_val
                combined_forecast["ensemble"]["method"] = "ARIMA only"
            
            combined.append(combined_forecast)
        
        return combined
    
    def _detect_anomalies(self, forecast_data: List[Dict]) -> List[Dict[str, Any]]:
        """Detect anomalies in the forecast data"""
        anomalies = []
        
        try:
            if not forecast_data:
                return anomalies
            
            # Get ensemble forecast values
            ensemble_values = []
            for item in forecast_data:
                if "ensemble" in item and "yhat" in item["ensemble"]:
                    ensemble_values.append(item["ensemble"]["yhat"])
            
            if len(ensemble_values) < 2:
                return anomalies
            
            # Calculate statistics
            mean_val = np.mean(ensemble_values)
            std_val = np.std(ensemble_values)
            
            # Define anomaly threshold (2 standard deviations)
            threshold = 2 * std_val
            
            # Detect anomalies
            for i, item in enumerate(forecast_data):
                if "ensemble" in item and "yhat" in item["ensemble"]:
                    value = item["ensemble"]["yhat"]
                    deviation = abs(value - mean_val)
                    
                    if deviation > threshold:
                        anomaly = {
                            "period": item["period"],
                            "value": value,
                            "expected": mean_val,
                            "deviation": deviation,
                            "severity": "High" if deviation > 3 * std_val else "Medium"
                        }
                        anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return []
    
    def _evaluate_models(self) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            if self.data is None or len(self.data) < 20:
                return {"error": "Insufficient data for evaluation"}
            
            # Split data for evaluation
            split_point = int(len(self.data) * 0.8)
            train_data = self.data.iloc[:split_point]
            test_data = self.data.iloc[split_point:]
            
            # Calculate metrics
            metrics = {}
            
            if self.prophet_model:
                # Prophet evaluation would go here
                metrics["prophet"] = {"status": "Trained"}
            
            if self.arima_model:
                # ARIMA evaluation would go here
                metrics["arima"] = {"status": "Trained"}
            
            return metrics
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the data"""
        try:
            if self.data is None:
                return {}
            
            return {
                "total_points": len(self.data),
                "date_range": {
                    "start": str(self.data.index[0]),
                    "end": str(self.data.index[-1])
                },
                "statistics": {
                    "mean": float(self.data['value'].mean()),
                    "std": float(self.data['value'].std()),
                    "min": float(self.data['value'].min()),
                    "max": float(self.data['value'].max())
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_forecast_plot(self) -> Dict[str, Any]:
        """Generate plot data for visualization"""
        try:
            if self.forecast_data is None:
                return {"error": "No forecast data available"}
            
            # This would return plot data for Streamlit visualization
            return {
                "data_points": len(self.data) if self.data is not None else 0,
                "forecast_points": len(self.forecast_data),
                "anomaly_count": len(self.anomalies)
            }
            
        except Exception as e:
            return {"error": str(e)}
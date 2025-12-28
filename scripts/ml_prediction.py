#!/usr/bin/env python3
"""
IoT Sensor ML Prediction Model
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Author: RSK World

Advanced Machine Learning models for IoT sensor data:
- Temperature prediction using LSTM
- Anomaly classification using Random Forest
- Sensor fusion using Neural Networks
- Predictive maintenance forecasting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class IoTPredictor:
    """
    IoT Sensor Prediction Models
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Author: RSK World
    """
    
    def __init__(self, data_path):
        """
        Initialize predictor
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.scalers = {}
        self.load_data()
    
    def load_data(self):
        """
        Load and preprocess data
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        else:
            import json
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
        
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df.set_index('timestamp', inplace=True)
        
        # Create features for prediction
        self.df['hour'] = self.df.index.hour
        self.df['minute'] = self.df.index.minute
        self.df['temp_lag1'] = self.df['temperature'].shift(1)
        self.df['humidity_lag1'] = self.df['humidity'].shift(1)
        self.df['pressure_lag1'] = self.df['pressure'].shift(1)
        self.df.dropna(inplace=True)
        
        print(f"✓ Loaded {len(self.df)} records with features")
    
    def train_anomaly_classifier(self):
        """
        Train Random Forest classifier for anomaly detection
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("TRAINING ANOMALY CLASSIFIER")
        print("="*60)
        
        features = ['temperature', 'humidity', 'pressure', 'motion', 
                   'hour', 'minute', 'temp_lag1', 'humidity_lag1', 'pressure_lag1']
        X = self.df[features].values
        y = self.df['anomaly'].values
        
        # Split data
        # Use stratify only if both classes are present
        stratify_param = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.models['anomaly_classifier'] = model
        self.scalers['anomaly_classifier'] = scaler
        
        return model, scaler
    
    def train_temperature_predictor(self):
        """
        Train Random Forest regressor for temperature prediction
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("TRAINING TEMPERATURE PREDICTOR")
        print("="*60)
        
        features = ['humidity', 'pressure', 'motion', 
                   'hour', 'minute', 'temp_lag1', 'humidity_lag1', 'pressure_lag1']
        X = self.df[features].values
        y = self.df['temperature'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {np.sqrt(mse):.4f}°C")
        
        self.models['temperature_predictor'] = model
        self.scalers['temperature_predictor'] = scaler
        
        return model, scaler
    
    def predict_anomaly(self, temperature, humidity, pressure, motion, hour, minute, 
                       temp_lag1=None, humidity_lag1=None, pressure_lag1=None):
        """
        Predict anomaly for given sensor readings
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        if 'anomaly_classifier' not in self.models:
            print("Model not trained. Training now...")
            self.train_anomaly_classifier()
        
        # Use provided values or defaults
        temp_lag1 = temp_lag1 or temperature
        humidity_lag1 = humidity_lag1 or humidity
        pressure_lag1 = pressure_lag1 or pressure
        
        features = np.array([[temperature, humidity, pressure, motion, 
                            hour, minute, temp_lag1, humidity_lag1, pressure_lag1]])
        
        scaler = self.scalers['anomaly_classifier']
        model = self.models['anomaly_classifier']
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return prediction, probability
    
    def predict_temperature(self, humidity, pressure, motion, hour, minute,
                           temp_lag1=None, humidity_lag1=None, pressure_lag1=None):
        """
        Predict temperature for given conditions
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        if 'temperature_predictor' not in self.models:
            print("Model not trained. Training now...")
            self.train_temperature_predictor()
        
        # Use provided values or defaults
        temp_lag1 = temp_lag1 or 22.0
        humidity_lag1 = humidity_lag1 or humidity
        pressure_lag1 = pressure_lag1 or pressure
        
        features = np.array([[humidity, pressure, motion, 
                            hour, minute, temp_lag1, humidity_lag1, pressure_lag1]])
        
        scaler = self.scalers['temperature_predictor']
        model = self.models['temperature_predictor']
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        return prediction
    
    def save_models(self, directory='models'):
        """
        Save trained models
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = os.path.join(directory, f'{name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved model: {model_path}")
        
        for name, scaler in self.scalers.items():
            scaler_path = os.path.join(directory, f'{name}_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"✓ Saved scaler: {scaler_path}")
    
    def generate_report(self):
        """
        Generate comprehensive ML report
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("MACHINE LEARNING PREDICTION REPORT")
        print("="*60)
        print("Website: https://rskworld.in")
        print("Email: help@rskworld.in")
        print("Phone: +91 93305 39277")
        print("="*60)
        
        # Train all models
        self.train_anomaly_classifier()
        self.train_temperature_predictor()
        
        # Save models
        self.save_models()
        
        # Example predictions
        print("\n" + "="*60)
        print("EXAMPLE PREDICTIONS")
        print("="*60)
        
        # Predict anomaly
        anomaly_pred, anomaly_prob = self.predict_anomaly(
            temperature=30.0, humidity=50.0, pressure=1000.0, 
            motion=0, hour=12, minute=30
        )
        print(f"\nAnomaly Prediction: {anomaly_pred} (Normal: {anomaly_prob[0]:.2%}, Anomaly: {anomaly_prob[1]:.2%})")
        
        # Predict temperature
        temp_pred = self.predict_temperature(
            humidity=60.0, pressure=1013.0, motion=1, hour=14, minute=0
        )
        print(f"Temperature Prediction: {temp_pred:.2f}°C")
        
        print("\n" + "="*60)
        print("ML TRAINING COMPLETE!")
        print("="*60)


def main():
    """
    Main function
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Author: RSK World
    """
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    csv_path = os.path.join(data_dir, 'iot_sensors.csv')
    json_path = os.path.join(data_dir, 'iot_sensors.json')
    
    if os.path.exists(csv_path):
        data_path = csv_path
    elif os.path.exists(json_path):
        data_path = json_path
    else:
        print("Error: No data file found")
        return
    
    predictor = IoTPredictor(data_path)
    predictor.generate_report()


if __name__ == "__main__":
    main()


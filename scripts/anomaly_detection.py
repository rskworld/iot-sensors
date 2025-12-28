#!/usr/bin/env python3
"""
IoT Sensor Anomaly Detection Script
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Author: RSK World

This script implements anomaly detection algorithms for IoT sensor data:
- Statistical outlier detection (Z-score method)
- Isolation Forest
- Local Outlier Factor (LOF)
- Visualization of detected anomalies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class AnomalyDetector:
    """
    Anomaly Detection Class for IoT Sensor Data
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Author: RSK World
    """
    
    def __init__(self, data_path):
        """
        Initialize the anomaly detector
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """
        Load sensor data
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
        print(f"✓ Loaded {len(self.df)} records")
    
    def zscore_detection(self, threshold=3):
        """
        Detect anomalies using Z-score method
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("Z-SCORE ANOMALY DETECTION")
        print("="*60)
        
        features = ['temperature', 'humidity', 'pressure']
        anomalies = pd.Series([0] * len(self.df), index=self.df.index)
        
        for feature in features:
            z_scores = np.abs((self.df[feature] - self.df[feature].mean()) / self.df[feature].std())
            anomalies = anomalies | (z_scores > threshold).astype(int)
        
        self.df['zscore_anomaly'] = anomalies
        detected = anomalies.sum()
        print(f"Detected {detected} anomalies ({detected/len(self.df)*100:.2f}%)")
        
        return anomalies
    
    def isolation_forest_detection(self, contamination=0.1):
        """
        Detect anomalies using Isolation Forest
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("ISOLATION FOREST ANOMALY DETECTION")
        print("="*60)
        
        # Prepare features
        features = ['temperature', 'humidity', 'pressure']
        X = self.df[features].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(X_scaled)
        
        # Convert predictions: -1 = anomaly, 1 = normal
        anomalies = (predictions == -1).astype(int)
        self.df['isolation_forest_anomaly'] = anomalies
        
        detected = anomalies.sum()
        print(f"Detected {detected} anomalies ({detected/len(self.df)*100:.2f}%)")
        
        return anomalies
    
    def lof_detection(self, n_neighbors=20, contamination=0.1):
        """
        Detect anomalies using Local Outlier Factor
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("LOCAL OUTLIER FACTOR (LOF) ANOMALY DETECTION")
        print("="*60)
        
        # Prepare features
        features = ['temperature', 'humidity', 'pressure']
        X = self.df[features].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit LOF
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        predictions = lof.fit_predict(X_scaled)
        
        # Convert predictions: -1 = anomaly, 1 = normal
        anomalies = (predictions == -1).astype(int)
        self.df['lof_anomaly'] = anomalies
        
        detected = anomalies.sum()
        print(f"Detected {detected} anomalies ({detected/len(self.df)*100:.2f}%)")
        
        return anomalies
    
    def compare_methods(self):
        """
        Compare different anomaly detection methods
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("METHOD COMPARISON")
        print("="*60)
        
        methods = ['zscore_anomaly', 'isolation_forest_anomaly', 'lof_anomaly']
        if 'anomaly' in self.df.columns:
            methods.append('anomaly')
        
        comparison = pd.DataFrame()
        for method in methods:
            if method in self.df.columns:
                comparison[method] = self.df[method]
        
        print("\nAnomaly Detection Summary:")
        print(comparison.sum())
        
        # Agreement between methods
        if len(comparison.columns) >= 2:
            agreement = (comparison.sum(axis=1) >= 2).sum()
            print(f"\nAgreement (2+ methods): {agreement} anomalies")
    
    def visualize_anomalies(self):
        """
        Visualize detected anomalies
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Temperature with anomalies
        axes[0].plot(self.df.index, self.df['temperature'], 
                    color='blue', alpha=0.6, linewidth=1.5, label='Temperature')
        
        # Mark different types of anomalies
        if 'zscore_anomaly' in self.df.columns:
            zscore_anoms = self.df[self.df['zscore_anomaly'] == 1]
            if len(zscore_anoms) > 0:
                axes[0].scatter(zscore_anoms.index, zscore_anoms['temperature'],
                               color='red', s=100, marker='x', 
                               label='Z-Score Anomalies', zorder=5)
        
        if 'isolation_forest_anomaly' in self.df.columns:
            iso_anoms = self.df[self.df['isolation_forest_anomaly'] == 1]
            if len(iso_anoms) > 0:
                axes[0].scatter(iso_anoms.index, iso_anoms['temperature'],
                               color='orange', s=80, marker='o', 
                               label='Isolation Forest', zorder=5, alpha=0.7)
        
        axes[0].set_title('Temperature with Detected Anomalies', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Humidity with anomalies
        axes[1].plot(self.df.index, self.df['humidity'], 
                    color='green', alpha=0.6, linewidth=1.5, label='Humidity')
        
        if 'zscore_anomaly' in self.df.columns:
            zscore_anoms = self.df[self.df['zscore_anomaly'] == 1]
            if len(zscore_anoms) > 0:
                axes[1].scatter(zscore_anoms.index, zscore_anoms['humidity'],
                               color='red', s=100, marker='x', 
                               label='Z-Score Anomalies', zorder=5)
        
        axes[1].set_title('Humidity with Detected Anomalies', 
                         fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Humidity (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Pressure with anomalies
        axes[2].plot(self.df.index, self.df['pressure'], 
                    color='purple', alpha=0.6, linewidth=1.5, label='Pressure')
        
        if 'zscore_anomaly' in self.df.columns:
            zscore_anoms = self.df[self.df['zscore_anomaly'] == 1]
            if len(zscore_anoms) > 0:
                axes[2].scatter(zscore_anoms.index, zscore_anoms['pressure'],
                               color='red', s=100, marker='x', 
                               label='Z-Score Anomalies', zorder=5)
        
        axes[2].set_title('Pressure with Detected Anomalies', 
                         fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Timestamp')
        axes[2].set_ylabel('Pressure (hPa)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved visualization: anomaly_detection_results.png")
        plt.close()
    
    def generate_report(self):
        """
        Generate comprehensive anomaly detection report
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("ANOMALY DETECTION REPORT")
        print("="*60)
        print("Website: https://rskworld.in")
        print("Email: help@rskworld.in")
        print("Phone: +91 93305 39277")
        print("="*60)
        
        # Run all detection methods
        self.zscore_detection()
        self.isolation_forest_detection()
        self.lof_detection()
        
        # Compare methods
        self.compare_methods()
        
        # Visualize
        self.visualize_anomalies()
        
        print("\n" + "="*60)
        print("ANOMALY DETECTION COMPLETE!")
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
    
    # Determine data file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    csv_path = os.path.join(data_dir, 'iot_sensors.csv')
    json_path = os.path.join(data_dir, 'iot_sensors.json')
    
    # Use CSV if available, otherwise JSON
    if os.path.exists(csv_path):
        data_path = csv_path
    elif os.path.exists(json_path):
        data_path = json_path
    else:
        print("Error: No data file found in data/ directory")
        return
    
    # Initialize detector and run analysis
    detector = AnomalyDetector(data_path)
    detector.generate_report()


if __name__ == "__main__":
    main()


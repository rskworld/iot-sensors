#!/usr/bin/env python3
"""
IoT Sensor Dataset Analysis Script
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Author: RSK World

This script performs comprehensive analysis on IoT sensor data including:
- Data loading and preprocessing
- Statistical analysis
- Temporal pattern analysis
- Device-wise comparisons
- Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class IoTSensorAnalyzer:
    """
    IoT Sensor Data Analyzer Class
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Author: RSK World
    """
    
    def __init__(self, data_path):
        """
        Initialize the analyzer with data path
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
        Load data from CSV or JSON file
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.json'):
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df.set_index('timestamp', inplace=True)
        print(f"✓ Loaded {len(self.df)} records from {self.data_path}")
    
    def basic_statistics(self):
        """
        Calculate basic statistics for all sensors
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)
        print(self.df[['temperature', 'humidity', 'pressure']].describe())
        
        print("\n" + "="*60)
        print("ANOMALY SUMMARY")
        print("="*60)
        anomaly_count = self.df['anomaly'].sum()
        total_count = len(self.df)
        print(f"Total Anomalies: {anomaly_count} ({anomaly_count/total_count*100:.2f}%)")
        print(f"Normal Readings: {total_count - anomaly_count} ({(total_count-anomaly_count)/total_count*100:.2f}%)")
        
        print("\n" + "="*60)
        print("DEVICE SUMMARY")
        print("="*60)
        device_stats = self.df.groupby('device_id').agg({
            'temperature': ['mean', 'std', 'min', 'max'],
            'humidity': ['mean', 'std', 'min', 'max'],
            'pressure': ['mean', 'std', 'min', 'max'],
            'motion': 'sum',
            'anomaly': 'sum'
        })
        print(device_stats)
    
    def temporal_analysis(self):
        """
        Analyze temporal patterns in the data
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("TEMPORAL ANALYSIS")
        print("="*60)
        
        # Resample by hour
        hourly = self.df.resample('H').agg({
            'temperature': 'mean',
            'humidity': 'mean',
            'pressure': 'mean',
            'motion': 'sum',
            'anomaly': 'sum'
        })
        
        print("\nHourly Averages:")
        print(hourly.head(10))
        
        return hourly
    
    def visualize_temperature_humidity(self):
        """
        Create visualization for temperature and humidity
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Temperature over time
        ax1.plot(self.df.index, self.df['temperature'], 
                color='red', alpha=0.7, linewidth=1.5)
        ax1.set_title('Temperature Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Timestamp')
        ax1.set_ylabel('Temperature (°C)')
        ax1.grid(True, alpha=0.3)
        
        # Highlight anomalies
        anomalies = self.df[self.df['anomaly'] == 1]
        if len(anomalies) > 0:
            ax1.scatter(anomalies.index, anomalies['temperature'], 
                       color='red', s=100, marker='x', 
                       label='Anomalies', zorder=5)
            ax1.legend()
        
        # Humidity over time
        ax2.plot(self.df.index, self.df['humidity'], 
                color='blue', alpha=0.7, linewidth=1.5)
        ax2.set_title('Humidity Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Humidity (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('temperature_humidity_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved visualization: temperature_humidity_analysis.png")
        plt.close()
    
    def visualize_device_comparison(self):
        """
        Compare sensor readings across different devices
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Temperature by device
        self.df.boxplot(column='temperature', by='device_id', ax=axes[0, 0])
        axes[0, 0].set_title('Temperature Distribution by Device')
        axes[0, 0].set_xlabel('Device ID')
        axes[0, 0].set_ylabel('Temperature (°C)')
        
        # Humidity by device
        self.df.boxplot(column='humidity', by='device_id', ax=axes[0, 1])
        axes[0, 1].set_title('Humidity Distribution by Device')
        axes[0, 1].set_xlabel('Device ID')
        axes[0, 1].set_ylabel('Humidity (%)')
        
        # Pressure by device
        self.df.boxplot(column='pressure', by='device_id', ax=axes[1, 0])
        axes[1, 0].set_title('Pressure Distribution by Device')
        axes[1, 0].set_xlabel('Device ID')
        axes[1, 0].set_ylabel('Pressure (hPa)')
        
        # Motion detection by device
        motion_by_device = self.df.groupby('device_id')['motion'].sum()
        axes[1, 1].bar(motion_by_device.index, motion_by_device.values, 
                      color='green', alpha=0.7)
        axes[1, 1].set_title('Motion Detection Count by Device')
        axes[1, 1].set_xlabel('Device ID')
        axes[1, 1].set_ylabel('Motion Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Device Comparison Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('device_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved visualization: device_comparison.png")
        plt.close()
    
    def correlation_analysis(self):
        """
        Analyze correlations between different sensors
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        corr_matrix = self.df[['temperature', 'humidity', 'pressure']].corr()
        print(corr_matrix)
        
        # Visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=1, 
                   cbar_kws={"shrink": 0.8})
        plt.title('Sensor Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved visualization: correlation_matrix.png")
        plt.close()
    
    def generate_report(self):
        """
        Generate comprehensive analysis report
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("GENERATING ANALYSIS REPORT")
        print("="*60)
        
        self.basic_statistics()
        self.temporal_analysis()
        self.correlation_analysis()
        self.visualize_temperature_humidity()
        self.visualize_device_comparison()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("  - temperature_humidity_analysis.png")
        print("  - device_comparison.png")
        print("  - correlation_matrix.png")
        print("\nWebsite: https://rskworld.in")
        print("Email: help@rskworld.in")
        print("Phone: +91 93305 39277")


def main():
    """
    Main function to run the analysis
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Author: RSK World
    """
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
    
    # Initialize analyzer and run analysis
    analyzer = IoTSensorAnalyzer(data_path)
    analyzer.generate_report()


if __name__ == "__main__":
    main()


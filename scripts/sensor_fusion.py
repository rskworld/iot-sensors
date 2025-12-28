#!/usr/bin/env python3
"""
IoT Sensor Fusion Algorithm
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Author: RSK World

Advanced sensor fusion techniques:
- Kalman Filter for sensor fusion
- Weighted average fusion
- Bayesian fusion
- Multi-sensor data integration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

class SensorFusion:
    """
    Sensor Fusion Algorithms for IoT Data
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Author: RSK World
    """
    
    def __init__(self, data_path):
        """
        Initialize sensor fusion
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
    
    def weighted_average_fusion(self, weights=None):
        """
        Weighted average fusion of multiple sensors
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("WEIGHTED AVERAGE SENSOR FUSION")
        print("="*60)
        
        if weights is None:
            # Default weights based on sensor reliability
            weights = {
                'temperature': 0.4,
                'humidity': 0.3,
                'pressure': 0.3
            }
        
        # Normalize sensors to 0-1 scale for fusion
        temp_range = self.df['temperature'].max() - self.df['temperature'].min()
        temp_norm = (self.df['temperature'] - self.df['temperature'].min()) / \
                   (temp_range if temp_range > 0 else 1)
        
        hum_range = self.df['humidity'].max() - self.df['humidity'].min()
        hum_norm = (self.df['humidity'] - self.df['humidity'].min()) / \
                  (hum_range if hum_range > 0 else 1)
        
        press_range = self.df['pressure'].max() - self.df['pressure'].min()
        press_norm = (self.df['pressure'] - self.df['pressure'].min()) / \
                    (press_range if press_range > 0 else 1)
        
        # Weighted fusion
        fused = (weights['temperature'] * temp_norm + 
                weights['humidity'] * hum_norm + 
                weights['pressure'] * press_norm)
        
        self.df['fused_signal'] = fused
        
        print(f"Fused signal range: {fused.min():.4f} to {fused.max():.4f}")
        print(f"Mean fused value: {fused.mean():.4f}")
        
        return fused
    
    def kalman_filter_fusion(self):
        """
        Kalman Filter for sensor fusion
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("KALMAN FILTER SENSOR FUSION")
        print("="*60)
        
        # Simple Kalman Filter implementation
        measurements = self.df[['temperature', 'humidity', 'pressure']].values
        
        # Initial state
        x = np.array([measurements[0].mean()])  # State estimate
        P = 1.0  # Estimate uncertainty
        Q = 0.01  # Process noise
        R = 0.1  # Measurement noise
        
        kalman_estimates = []
        
        for z in measurements:
            # Prediction step
            x_pred = x
            P_pred = P + Q
            
            # Update step (fuse multiple measurements)
            z_mean = z.mean()
            K = P_pred / (P_pred + R)  # Kalman gain
            x = x_pred + K * (z_mean - x_pred)
            P = (1 - K) * P_pred
            
            kalman_estimates.append(x[0])
        
        self.df['kalman_fused'] = kalman_estimates
        
        print(f"Kalman filtered signal range: {min(kalman_estimates):.4f} to {max(kalman_estimates):.4f}")
        print(f"Mean Kalman value: {np.mean(kalman_estimates):.4f}")
        
        return np.array(kalman_estimates)
    
    def bayesian_fusion(self):
        """
        Bayesian sensor fusion
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("BAYESIAN SENSOR FUSION")
        print("="*60)
        
        # Calculate statistics for each sensor
        temp_mean = self.df['temperature'].mean()
        temp_std = self.df['temperature'].std()
        hum_mean = self.df['humidity'].mean()
        hum_std = self.df['humidity'].std()
        press_mean = self.df['pressure'].mean()
        press_std = self.df['pressure'].std()
        
        # Bayesian fusion using weighted combination of distributions
        # Weight by inverse variance (more reliable sensors get higher weight)
        # Avoid division by zero
        temp_weight = 1 / (temp_std ** 2) if temp_std > 0 else 1.0
        hum_weight = 1 / (hum_std ** 2) if hum_std > 0 else 1.0
        press_weight = 1 / (press_std ** 2) if press_std > 0 else 1.0
        total_weight = temp_weight + hum_weight + press_weight
        
        # Normalized weights
        temp_weight_norm = temp_weight / total_weight
        hum_weight_norm = hum_weight / total_weight
        press_weight_norm = press_weight / total_weight
        
        # Fused estimate
        bayesian_fused = (temp_weight_norm * self.df['temperature'] + 
                         hum_weight_norm * self.df['humidity'] + 
                         press_weight_norm * self.df['pressure'])
        
        self.df['bayesian_fused'] = bayesian_fused
        
        print(f"Bayesian fused signal range: {bayesian_fused.min():.4f} to {bayesian_fused.max():.4f}")
        print(f"Mean Bayesian value: {bayesian_fused.mean():.4f}")
        print(f"\nSensor Weights:")
        print(f"  Temperature: {temp_weight_norm:.4f}")
        print(f"  Humidity: {hum_weight_norm:.4f}")
        print(f"  Pressure: {press_weight_norm:.4f}")
        
        return bayesian_fused
    
    def visualize_fusion(self):
        """
        Visualize sensor fusion results
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Original sensors
        axes[0, 0].plot(self.df.index, self.df['temperature'], 
                       label='Temperature', alpha=0.7, linewidth=1.5)
        axes[0, 0].plot(self.df.index, self.df['humidity']/10, 
                       label='Humidity/10', alpha=0.7, linewidth=1.5)
        axes[0, 0].plot(self.df.index, self.df['pressure']/50, 
                       label='Pressure/50', alpha=0.7, linewidth=1.5)
        axes[0, 0].set_title('Original Sensor Readings', fontweight='bold')
        axes[0, 0].set_xlabel('Timestamp')
        axes[0, 0].set_ylabel('Normalized Values')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Weighted average fusion
        if 'fused_signal' in self.df.columns:
            axes[0, 1].plot(self.df.index, self.df['fused_signal'], 
                          color='red', linewidth=2, label='Fused Signal')
            axes[0, 1].set_title('Weighted Average Fusion', fontweight='bold')
            axes[0, 1].set_xlabel('Timestamp')
            axes[0, 1].set_ylabel('Fused Value')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Kalman filter fusion
        if 'kalman_fused' in self.df.columns:
            axes[1, 0].plot(self.df.index, self.df['kalman_fused'], 
                          color='green', linewidth=2, label='Kalman Filtered')
            axes[1, 0].set_title('Kalman Filter Fusion', fontweight='bold')
            axes[1, 0].set_xlabel('Timestamp')
            axes[1, 0].set_ylabel('Fused Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Bayesian fusion
        if 'bayesian_fused' in self.df.columns:
            axes[1, 1].plot(self.df.index, self.df['bayesian_fused'], 
                          color='purple', linewidth=2, label='Bayesian Fused')
            axes[1, 1].set_title('Bayesian Fusion', fontweight='bold')
            axes[1, 1].set_xlabel('Timestamp')
            axes[1, 1].set_ylabel('Fused Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Sensor Fusion Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('sensor_fusion_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved visualization: sensor_fusion_analysis.png")
        plt.close()
    
    def generate_report(self):
        """
        Generate comprehensive fusion report
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("SENSOR FUSION REPORT")
        print("="*60)
        print("Website: https://rskworld.in")
        print("Email: help@rskworld.in")
        print("Phone: +91 93305 39277")
        print("="*60)
        
        # Run all fusion methods
        self.weighted_average_fusion()
        self.kalman_filter_fusion()
        self.bayesian_fusion()
        
        # Visualize
        self.visualize_fusion()
        
        print("\n" + "="*60)
        print("SENSOR FUSION COMPLETE!")
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
    
    fusion = SensorFusion(data_path)
    fusion.generate_report()


if __name__ == "__main__":
    main()


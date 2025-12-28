#!/usr/bin/env python3
"""
IoT Real-Time Data Simulator
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Author: RSK World

Simulates real-time IoT sensor data streaming:
- Live data generation
- WebSocket-like streaming simulation
- Real-time visualization
- Data export to CSV/JSON
"""

import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class RealTimeSimulator:
    """
    Real-Time IoT Sensor Data Simulator
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Author: RSK World
    """
    
    def __init__(self, devices=['DEV001', 'DEV002', 'DEV003', 'DEV004', 'DEV005']):
        """
        Initialize simulator
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        self.devices = devices
        self.data_buffer = []
        self.running = False
    
    def generate_sensor_reading(self, device_id, base_temp=22.0, base_humidity=60.0, 
                                base_pressure=1013.0):
        """
        Generate a single sensor reading
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        # Add some realistic variation
        temp = base_temp + np.random.normal(0, 2) + np.sin(time.time() / 3600) * 3
        humidity = max(20, min(90, base_humidity + np.random.normal(0, 5)))
        pressure = base_pressure + np.random.normal(0, 2)
        motion = 1 if np.random.random() > 0.7 else 0
        
        # Detect anomalies
        anomaly = 1 if (temp > 32 or humidity < 25 or pressure < 1000) else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'device_id': device_id,
            'temperature': round(temp, 2),
            'humidity': round(humidity, 2),
            'pressure': round(pressure, 2),
            'motion': motion,
            'anomaly': anomaly
        }
    
    def stream_data(self, duration=60, interval=1, output_file=None):
        """
        Stream sensor data in real-time
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print("\n" + "="*60)
        print("REAL-TIME SENSOR DATA STREAMING")
        print("="*60)
        print(f"Duration: {duration} seconds")
        print(f"Interval: {interval} seconds")
        print("Press Ctrl+C to stop\n")
        
        self.running = True
        start_time = time.time()
        count = 0
        
        try:
            while self.running and (time.time() - start_time) < duration:
                # Generate reading for each device
                for device in self.devices:
                    reading = self.generate_sensor_reading(device)
                    self.data_buffer.append(reading)
                    count += 1
                    
                    # Print to console
                    print(f"[{reading['timestamp']}] {device}: "
                          f"Temp={reading['temperature']}°C, "
                          f"Humidity={reading['humidity']}%, "
                          f"Pressure={reading['pressure']}hPa, "
                          f"Motion={reading['motion']}, "
                          f"Anomaly={reading['anomaly']}")
                
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\nStreaming stopped by user")
        
        self.running = False
        print(f"\n✓ Generated {count} readings from {len(self.devices)} devices")
        
        # Save to file if specified
        if output_file:
            self.save_data(output_file)
    
    def save_data(self, filename):
        """
        Save buffered data to file
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        if not self.data_buffer:
            print("No data to save")
            return
        
        df = pd.DataFrame(self.data_buffer)
        
        if filename.endswith('.csv'):
            df.to_csv(filename, index=False)
            print(f"✓ Saved {len(df)} records to {filename}")
        elif filename.endswith('.json'):
            df.to_json(filename, orient='records', date_format='iso', indent=2)
            print(f"✓ Saved {len(df)} records to {filename}")
        else:
            print(f"Unsupported file format: {filename}")
    
    def generate_batch(self, num_readings=100, output_file=None):
        """
        Generate a batch of sensor readings
        Website: https://rskworld.in
        Email: help@rskworld.in
        Phone: +91 93305 39277
        Author: RSK World
        """
        print(f"\nGenerating {num_readings} sensor readings...")
        
        base_time = datetime.now()
        for i in range(num_readings):
            device = np.random.choice(self.devices)
            timestamp = base_time + timedelta(minutes=i)
            
            reading = self.generate_sensor_reading(device)
            reading['timestamp'] = timestamp.isoformat()
            self.data_buffer.append(reading)
        
        print(f"✓ Generated {len(self.data_buffer)} readings")
        
        if output_file:
            self.save_data(output_file)
        
        return pd.DataFrame(self.data_buffer)


def main():
    """
    Main function
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Author: RSK World
    """
    import sys
    
    simulator = RealTimeSimulator()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == 'stream':
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            interval = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
            simulator.stream_data(duration=duration, interval=interval)
        elif mode == 'batch':
            num = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            output = sys.argv[3] if len(sys.argv) > 3 else 'real_time_data.csv'
            simulator.generate_batch(num_readings=num, output_file=output)
        else:
            print("Usage: python real_time_simulator.py [stream|batch] [args...]")
    else:
        # Default: generate batch
        simulator.generate_batch(num_readings=50, output_file='real_time_data.csv')
        print("\n" + "="*60)
        print("REAL-TIME SIMULATOR")
        print("="*60)
        print("Website: https://rskworld.in")
        print("Email: help@rskworld.in")
        print("Phone: +91 93305 39277")
        print("="*60)


if __name__ == "__main__":
    main()


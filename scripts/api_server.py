#!/usr/bin/env python3
"""
IoT Sensor Data API Server
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Author: RSK World

RESTful API for IoT sensor data:
- GET /api/data - Retrieve sensor data
- GET /api/data/{device_id} - Get data for specific device
- GET /api/stats - Get statistics
- POST /api/data - Add new sensor reading
- GET /api/anomalies - Get anomaly data
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global data storage
sensor_data = []

def load_initial_data():
    """
    Load initial data from files
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Author: RSK World
    """
    global sensor_data
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    csv_path = os.path.join(data_dir, 'iot_sensors.csv')
    json_path = os.path.join(data_dir, 'iot_sensors.json')
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        sensor_data = df.to_dict('records')
    elif os.path.exists(json_path):
        with open(json_path, 'r') as f:
            sensor_data = json.load(f)
    
    print(f"✓ Loaded {len(sensor_data)} initial records")

@app.route('/')
def index():
    """
    API information endpoint
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Author: RSK World
    """
    return jsonify({
        'name': 'IoT Sensor Data API',
        'version': '1.0.0',
        'endpoints': {
            'GET /api/data': 'Retrieve all sensor data',
            'GET /api/data/<device_id>': 'Get data for specific device',
            'GET /api/stats': 'Get statistics',
            'POST /api/data': 'Add new sensor reading',
            'GET /api/anomalies': 'Get anomaly data'
        },
        'author': 'RSK World',
        'website': 'https://rskworld.in',
        'email': 'help@rskworld.in',
        'phone': '+91 93305 39277'
    })

@app.route('/api/data', methods=['GET'])
def get_data():
    """
    Get all sensor data
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Author: RSK World
    """
    limit = request.args.get('limit', type=int)
    device_id = request.args.get('device_id')
    
    data = sensor_data.copy()
    
    if device_id:
        data = [d for d in data if d.get('device_id') == device_id]
    
    if limit:
        data = data[:limit]
    
    return jsonify({
        'count': len(data),
        'data': data
    })

@app.route('/api/data/<device_id>', methods=['GET'])
def get_device_data(device_id):
    """
    Get data for specific device
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Author: RSK World
    """
    device_data = [d for d in sensor_data if d.get('device_id') == device_id]
    
    if not device_data:
        return jsonify({'error': 'Device not found'}), 404
    
    return jsonify({
        'device_id': device_id,
        'count': len(device_data),
        'data': device_data
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get statistics
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Author: RSK World
    """
    if not sensor_data:
        return jsonify({'error': 'No data available'}), 404
    
    df = pd.DataFrame(sensor_data)
    
    stats = {
        'total_readings': len(df),
        'devices': df['device_id'].unique().tolist(),
        'device_count': df['device_id'].nunique(),
        'temperature': {
            'mean': float(df['temperature'].mean()),
            'min': float(df['temperature'].min()),
            'max': float(df['temperature'].max()),
            'std': float(df['temperature'].std())
        },
        'humidity': {
            'mean': float(df['humidity'].mean()),
            'min': float(df['humidity'].min()),
            'max': float(df['humidity'].max()),
            'std': float(df['humidity'].std())
        },
        'pressure': {
            'mean': float(df['pressure'].mean()),
            'min': float(df['pressure'].min()),
            'max': float(df['pressure'].max()),
            'std': float(df['pressure'].std())
        },
        'anomalies': {
            'total': int(df['anomaly'].sum()),
            'percentage': float(df['anomaly'].mean() * 100)
        },
        'motion_detections': int(df['motion'].sum())
    }
    
    return jsonify(stats)

@app.route('/api/data', methods=['POST'])
def add_data():
    """
    Add new sensor reading
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Author: RSK World
    """
    data = request.get_json()
    
    required_fields = ['device_id', 'temperature', 'humidity', 'pressure']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Add timestamp if not provided
    if 'timestamp' not in data:
        data['timestamp'] = datetime.now().isoformat()
    
    # Add motion and anomaly if not provided
    if 'motion' not in data:
        data['motion'] = 0
    if 'anomaly' not in data:
        data['anomaly'] = 1 if (data['temperature'] > 32 or 
                               data['humidity'] < 25 or 
                               data['pressure'] < 1000) else 0
    
    sensor_data.append(data)
    
    return jsonify({
        'message': 'Data added successfully',
        'data': data
    }), 201

@app.route('/api/anomalies', methods=['GET'])
def get_anomalies():
    """
    Get anomaly data
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Author: RSK World
    """
    anomalies = [d for d in sensor_data if d.get('anomaly') == 1]
    
    return jsonify({
        'count': len(anomalies),
        'data': anomalies
    })

if __name__ == '__main__':
    load_initial_data()
    print("\n" + "="*60)
    print("IoT SENSOR DATA API SERVER")
    print("="*60)
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("Phone: +91 93305 39277")
    print("="*60)
    print("\nAPI Server running on http://localhost:5000")
    print("Available endpoints:")
    print("  GET  /api/data")
    print("  GET  /api/data/<device_id>")
    print("  GET  /api/stats")
    print("  POST /api/data")
    print("  GET  /api/anomalies")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)


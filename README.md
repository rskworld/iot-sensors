# IoT Sensor Dataset

<!--
    IoT Sensor Dataset - README
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
    Author: RSK World
-->

## Overview

This dataset contains multi-sensor IoT data with temperature, humidity, pressure, motion, and other sensor readings over time. Perfect for IoT analytics, anomaly detection, predictive maintenance, and sensor fusion applications.

**Website:** [https://rskworld.in](https://rskworld.in)  
**Email:** help@rskworld.in  
**Phone:** +91 93305 39277  
**Author:** RSK World

## Dataset Information

- **Category:** Time Series Data
- **Difficulty:** Advanced
- **Technologies:** CSV, JSON, Pandas, Time Series
- **Source:** RSK World - Free Programming Resources & Source Code

## Features

- ✅ Multi-sensor data (Temperature, Humidity, Pressure, Motion)
- ✅ Temporal patterns over time
- ✅ Anomaly labels for detection tasks
- ✅ Multiple devices (DEV001-DEV005)
- ✅ Sensor fusion ready
- ✅ CSV and JSON formats available

## Dataset Structure

### Columns

- `timestamp`: ISO format timestamp of the reading
- `device_id`: Unique identifier for the IoT device (DEV001-DEV005)
- `temperature`: Temperature reading in Celsius (°C)
- `humidity`: Humidity reading in percentage (%)
- `pressure`: Atmospheric pressure in hectopascals (hPa)
- `motion`: Binary indicator (0 = no motion, 1 = motion detected)
- `anomaly`: Binary label (0 = normal, 1 = anomaly detected)

### Sample Data

```csv
timestamp,device_id,temperature,humidity,pressure,motion,anomaly
2026-01-01 00:00:00,DEV001,22.5,65.3,1013.2,0,0
2026-01-01 00:01:00,DEV001,22.7,65.1,1013.4,0,0
```

## Files Included

```
iot-sensors/
├── index.html                  # Interactive demo page
├── README.md                   # This file
├── PROJECT_INFO.md             # Project details
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── iot-sensors.zip            # Complete project ZIP file
├── data/
│   ├── iot_sensors.csv         # CSV format dataset
│   └── iot_sensors.json        # JSON format dataset
└── scripts/
    ├── analyze_sensors.py          # Comprehensive data analysis
    ├── anomaly_detection.py        # Anomaly detection algorithms
    ├── ml_prediction.py            # Machine Learning models
    ├── sensor_fusion.py            # Sensor fusion algorithms
    ├── real_time_simulator.py      # Real-time data simulator
    └── api_server.py               # RESTful API server
```

## Quick Start

### 1. View the Demo

Open `index.html` in your web browser to see interactive visualizations of the dataset.

### 2. Load Data in Python

```python
import pandas as pd

# Load CSV
df = pd.read_csv('data/iot_sensors.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Or load JSON
import json
with open('data/iot_sensors.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)
```

### 3. Run Analysis Scripts

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run comprehensive analysis
python scripts/analyze_sensors.py

# Run anomaly detection
python scripts/anomaly_detection.py
```

## Use Cases

### 1. Time Series Analysis
- Analyze temporal patterns in sensor readings
- Identify trends and seasonality
- Forecast future sensor values

### 2. Anomaly Detection
- Detect unusual sensor readings
- Identify device malfunctions
- Implement predictive maintenance
- Multiple detection algorithms (Z-Score, Isolation Forest, LOF)

### 3. Sensor Fusion
- Combine multiple sensor readings
- Improve accuracy through data fusion
- Cross-validate sensor measurements
- Kalman Filter and Bayesian fusion

### 4. Machine Learning
- Train classification models for anomaly detection
- Build regression models for temperature prediction
- Model persistence and deployment
- Feature engineering with lag variables

### 5. Real-Time Applications
- Simulate real-time data streaming
- Generate live sensor data
- Test real-time processing pipelines

### 6. API Development
- RESTful API for sensor data access
- Statistics and analytics endpoints
- Data ingestion endpoints
- Integration with web applications

### 7. Device Comparison
- Compare performance across devices
- Identify device-specific patterns
- Quality assurance and testing

## Analysis Scripts

### analyze_sensors.py

Comprehensive analysis script that includes:
- Basic statistics and summaries
- Temporal pattern analysis
- Device-wise comparisons
- Correlation analysis
- Visualization generation

**Usage:**
```bash
python scripts/analyze_sensors.py
```

**Output:**
- Statistical summaries in console
- `temperature_humidity_analysis.png`
- `device_comparison.png`
- `correlation_matrix.png`

### anomaly_detection.py

Advanced anomaly detection using multiple algorithms:
- Z-Score method
- Isolation Forest
- Local Outlier Factor (LOF)
- Method comparison and visualization

**Usage:**
```bash
python scripts/anomaly_detection.py
```

**Output:**
- Anomaly detection results in console
- `anomaly_detection_results.png`

### ml_prediction.py

Machine Learning models for IoT sensor data:
- Random Forest classifier for anomaly detection
- Random Forest regressor for temperature prediction
- Feature engineering with lag variables
- Model training, evaluation, and persistence
- Example predictions

**Usage:**
```bash
python scripts/ml_prediction.py
```

**Output:**
- Trained models saved in `models/` directory
- Model performance metrics
- Example predictions

### sensor_fusion.py

Advanced sensor fusion techniques:
- Weighted average fusion
- Kalman Filter fusion
- Bayesian sensor fusion
- Multi-sensor data integration
- Visualization of fusion results

**Usage:**
```bash
python scripts/sensor_fusion.py
```

**Output:**
- Fusion analysis results
- `sensor_fusion_analysis.png`

### real_time_simulator.py

Real-time IoT sensor data simulator:
- Live data generation
- Streaming simulation
- Batch data generation
- CSV/JSON export

**Usage:**
```bash
# Stream data for 60 seconds with 1-second intervals
python scripts/real_time_simulator.py stream 60 1.0

# Generate batch of 100 readings
python scripts/real_time_simulator.py batch 100 output.csv
```

### api_server.py

RESTful API server for IoT sensor data:
- GET /api/data - Retrieve all sensor data
- GET /api/data/{device_id} - Get data for specific device
- GET /api/stats - Get statistics
- POST /api/data - Add new sensor reading
- GET /api/anomalies - Get anomaly data

**Usage:**
```bash
# Install Flask first: pip install flask flask-cors
python scripts/api_server.py
```

**Access:**
- API runs on http://localhost:5000
- View API documentation at http://localhost:5000/

## Requirements

### Python Packages

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
scipy>=1.10.0
flask>=2.3.0
flask-cors>=4.0.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Data Statistics

- **Total Records:** 121 readings
- **Devices:** 5 (DEV001-DEV005)
- **Time Range:** 2 hours (1-minute intervals)
- **Anomaly Rate:** ~41% (for demonstration purposes)

## License

This dataset is provided by RSK World for educational and research purposes.

## Contact

**RSK World**  
Website: [https://rskworld.in](https://rskworld.in)  
Email: help@rskworld.in  
Phone: +91 93305 39277

## Acknowledgments

Created by RSK World - Free Programming Resources & Source Code

---

*For more datasets and resources, visit [rskworld.in](https://rskworld.in)*


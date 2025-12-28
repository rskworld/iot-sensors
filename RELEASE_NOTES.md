# Release Notes - IoT Sensor Dataset v1.0.0

**Release Date:** December 28, 2025  
**Website:** https://rskworld.in  
**Email:** help@rskworld.in  
**Phone:** +91 93305 39277  
**Author:** RSK World

## 🎉 Initial Release v1.0.0

This is the first official release of the IoT Sensor Dataset project - a comprehensive dataset and analysis toolkit for IoT sensor data analytics.

## ✨ Features

### 📊 Dataset
- **121 sensor readings** from 5 IoT devices (DEV001-DEV005)
- **Time series data** spanning 2 hours with 1-minute intervals
- **Multiple sensor types:** Temperature, Humidity, Pressure, Motion
- **Anomaly labels** for supervised learning
- **Dual formats:** CSV and JSON

### 🔬 Analysis Scripts
1. **analyze_sensors.py** - Comprehensive data analysis
   - Statistical summaries
   - Temporal pattern analysis
   - Device-wise comparisons
   - Correlation analysis
   - Visualization generation

2. **anomaly_detection.py** - Advanced anomaly detection
   - Z-Score method
   - Isolation Forest algorithm
   - Local Outlier Factor (LOF)
   - Method comparison

3. **ml_prediction.py** - Machine Learning models
   - Random Forest classifier for anomaly detection
   - Random Forest regressor for temperature prediction
   - Feature engineering with lag variables
   - Model persistence

4. **sensor_fusion.py** - Sensor fusion algorithms
   - Weighted average fusion
   - Kalman Filter fusion
   - Bayesian sensor fusion
   - Multi-sensor integration

5. **real_time_simulator.py** - Real-time data simulator
   - Live data generation
   - Streaming simulation
   - Batch data generation
   - CSV/JSON export

6. **api_server.py** - RESTful API server
   - GET /api/data - Retrieve sensor data
   - GET /api/data/{device_id} - Device-specific data
   - GET /api/stats - Statistics
   - POST /api/data - Add new readings
   - GET /api/anomalies - Anomaly data

### 🌐 Interactive Demo
- **index.html** - Beautiful web-based dashboard
  - Real-time visualizations using Chart.js
  - Statistics dashboard
  - Interactive charts
  - Data preview table
  - Download links

## 📦 What's Included

```
iot-sensors/
├── index.html                  # Interactive demo page
├── README.md                   # Complete documentation
├── PROJECT_INFO.md             # Project details
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── iot-sensors.zip            # Complete project ZIP
├── data/
│   ├── iot_sensors.csv         # CSV format dataset
│   └── iot_sensors.json        # JSON format dataset
└── scripts/
    ├── analyze_sensors.py      # Data analysis
    ├── anomaly_detection.py    # Anomaly detection
    ├── ml_prediction.py        # ML models
    ├── sensor_fusion.py        # Sensor fusion
    ├── real_time_simulator.py  # Real-time simulator
    └── api_server.py           # REST API server
```

## 🚀 Quick Start

1. **View Demo:**
   ```bash
   # Open index.html in your browser
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Analysis:**
   ```bash
   python scripts/analyze_sensors.py
   python scripts/anomaly_detection.py
   python scripts/ml_prediction.py
   python scripts/sensor_fusion.py
   ```

4. **Start API Server:**
   ```bash
   python scripts/api_server.py
   # Access at http://localhost:5000
   ```

## 📋 Requirements

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- scikit-learn >= 1.2.0
- scipy >= 1.10.0
- flask >= 2.3.0
- flask-cors >= 4.0.0

## 🎯 Use Cases

- Time Series Analysis
- Anomaly Detection
- Sensor Fusion
- Machine Learning
- Real-Time Applications
- API Development
- Device Comparison

## 📝 Documentation

- Complete README with usage instructions
- Project information file
- Inline code documentation
- API documentation

## 🔗 Links

- **Repository:** https://github.com/rskworld/iot-sensors
- **Website:** https://rskworld.in
- **Email:** help@rskworld.in
- **Phone:** +91 93305 39277

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Created by **RSK World** - Free Programming Resources & Source Code

---

**For more datasets and resources, visit [rskworld.in](https://rskworld.in)**


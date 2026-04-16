# IntelliTraffic Pro — Advanced Traffic Intelligence & Congestion Optimization

A production-grade, real-time traffic monitoring and optimization system using **infrared** and **ultrasonic sensors** with an **ESP32 Gateway**, applying **KNN Predictive Modeling** and **Dynamic Trajectory Variance Optimization**.

---

## ✨ New Premium Features
*   **Production-Ready Backend:** Hardened with `Waitress` WSGI server and structured logging.
*   **Ultra-Premium Dashboard:** Glassmorphic UI with animated data visualizations and live ML sync.
*   **Interactive Mapping:** Real-time geospatial visualization of sensor nodes using Leaflet.js.
*   **Digital Twin Intersection:** Animated SVG representation of live traffic flow and signal states.
*   **Intelligence Sandbox:** Manual scenario testing for ML model verification.
*   **Enterprise Reporting:** On-the-fly PDF and CSV export of traffic intelligence data.

---

## 📁 Project Structure

```
IntelliTrafficPro/
├── hardware/
│   ├── arduino_sensors.ino     ← IR + ultrasonic sensor reading
│   └── esp32_wifi.ino          ← WiFi gateway (forwards data to server)
├── backend/
│   ├── server.py               ← Production Flask API (Waitress)
│   ├── database.py             ← SQLite storage + CSV export
│   ├── logger.py               ← Centralized rotating file logging
│   └── config.py               ← Environment-based configuration
├── analysis/
│   ├── feature_engineering.py  ← Windowed traffic feature extraction
│   ├── clustering.py           ← K-Means & DBSCAN traffic state clustering
│   ├── prediction.py           ← 87.5% Accuracy KNN congestion prediction
│   └── optimizer.py            ← Proportional phase split optimization
├── dashboard/
│   └── index.html              ← Premium Glassmorphic Dashboard
├── logs/                       ← Persistent application logs
├── data/                       ← Generated models, plots, and datasets
└── start_production.bat        ← One-click production server bootstrap
```

---

## ⚙️ Hardware Setup

| Component       | Connection                        |
|-----------------|-----------------------------------|
| IR Sensor       | Digital Pin 2 → Arduino           |
| HC-SR04 Trig    | Pin 9 → Arduino                   |
| HC-SR04 Echo    | Pin 10 → Arduino                  |
| Arduino TX      | → ESP32 GPIO 16 (Serial2 RX)     |
| Common GND      | Arduino GND ↔ ESP32 GND           |

1. Flash `hardware/arduino_sensors.ino` onto your Arduino.
2. Update WiFi credentials in `hardware/esp32_wifi.ino` and flash onto ESP32.
3. Power both boards — data flows: **Sensors → Arduino → ESP32 → WiFi → Server**.

---

## 🚀 Deployment

### 1. One-Click Production Start (Windows)
Simply run `start_production.bat`. This will:
- Automatically install missing dependencies.
- Set the environment to `production`.
- Launch the high-concurrency **Waitress** WSGI server.

### 2. Manual Startup
```bash
pip install -r requirements.txt
cd backend
python server.py
```
Server runs on `http://localhost:5000`. Access the dashboard via `dashboard/index.html`.

---

## 📊 Analytics Engine

| Component            | Method                     | Purpose                              |
|----------------------|----------------------------|--------------------------------------|
| Feature Engineering  | Sliding window (30 samples)| Extract vehicle volume, speed, occupancy |
| Clustering           | K-Means, DBSCAN           | Baseline traffic state identification   |
| Prediction           | KNN (87.5% Accuracy)       | Real-time congestion forecasting    |
| Optimization         | Differential Evolution     | Dynamic green-light phase split logic |

---

## 📈 Output Artifacts (in `data/`)
The system persists intelligence into `data/` including confusion matrices, cluster plots, and the pre-trained KNN model pickle.

---

## 📄 License
Production-grade prototype for intelligent urban infrastructure.


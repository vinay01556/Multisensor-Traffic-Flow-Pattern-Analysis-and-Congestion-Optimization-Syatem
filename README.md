# IntelliTraffic Pro — Advanced Traffic Intelligence & Congestion Optimization

A production-grade, real-time traffic monitoring and optimization system using **infrared** and **ultrasonic sensors** with an **ESP32 Gateway**, applying **KNN Predictive Modeling** and **Dynamic Trajectory Variance Optimization**.

---

## ✨ Features
*   **FastAPI Backend:** High-performance ASGI server with auto-generated Swagger docs.
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
│   ├── server.py               ← FastAPI + Uvicorn ASGI server
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

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Backend Server
```bash
cd backend
python server.py
```
The FastAPI server starts on **`http://localhost:5000`**.

- **Swagger UI (Interactive API Docs):** [http://localhost:5000/docs](http://localhost:5000/docs)
- **ReDoc (API Reference):** [http://localhost:5000/redoc](http://localhost:5000/redoc)
- **Health Check:** [http://localhost:5000/api/status](http://localhost:5000/api/status)

### 3. Run the Frontend Dashboard
Open `dashboard/index.html` directly in your browser:

**Windows:**
```cmd
start dashboard\index.html
```

**macOS / Linux:**
```bash
open dashboard/index.html        # macOS
xdg-open dashboard/index.html   # Linux
```

Or use a local HTTP server for best results:
```bash
cd dashboard
python -m http.server 8080
```
Then open **`http://localhost:8080`** in your browser.

### 4. One-Click Production Start (Windows)
Simply double-click `start_production.bat`. This will:
- Automatically install missing dependencies.
- Launch the **Uvicorn** ASGI server on port 5000.

---

## 🔌 API Endpoints

| Method | Endpoint              | Description                          |
|--------|-----------------------|--------------------------------------|
| `GET`  | `/api/status`         | Health check & record count          |
| `POST` | `/api/sensor-data`    | Receive sensor reading from ESP32    |
| `GET`  | `/api/sensor-data`    | Get recent readings (`?limit=N`)     |
| `GET`  | `/api/sensor-data/all`| Get all readings for analysis        |
| `GET`  | `/api/predict`        | Auto-predict congestion from history |
| `POST` | `/api/predict/manual` | Predict from manual feature input    |

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

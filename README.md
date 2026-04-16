# CSE275 — Multisensor Traffic Flow Pattern Analysis & Congestion Optimization System

A real-time traffic monitoring and optimization system using **infrared** and **ultrasonic sensors** with **Arduino** and **ESP32**, applying **machine learning clustering** and **trajectory variance optimization**.

---

## 📁 Project Structure

```
CSE275 project/
├── hardware/
│   ├── arduino_sensors.ino     ← IR + ultrasonic sensor reading
│   └── esp32_wifi.ino          ← WiFi gateway (forwards data to server)
├── backend/
│   ├── server.py               ← Flask API server
│   └── database.py             ← SQLite storage + CSV export
├── analysis/
│   ├── feature_engineering.py  ← Windowed traffic feature extraction
│   ├── clustering.py           ← K-Means & DBSCAN traffic state clustering
│   ├── prediction.py           ← KNN & Decision Tree congestion prediction
│   └── optimizer.py            ← Trajectory variance minimization (scipy)
├── dashboard/
│   └── index.html              ← Real-time web dashboard
├── data/                       ← Generated data, models, plots
└── README.md
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

1. Flash `hardware/arduino_sensors.ino` onto your Arduino
2. Update WiFi credentials in `hardware/esp32_wifi.ino` and flash onto ESP32
3. Power both boards — data flows: **Sensors → Arduino → ESP32 → WiFi → Server**

---

## 🚀 Quick Start

### 1. Install Python Dependencies

```bash
pip install flask flask-cors pandas numpy scikit-learn scipy matplotlib
```

### 2. Start the Backend Server

```bash
cd backend
python server.py
```

Server runs on `http://localhost:5000`. The ESP32 should POST to `/api/sensor-data`.

### 3. Run the Analysis Pipeline (with synthetic data for testing)

```bash
cd analysis

# Step 1: Generate features (includes synthetic data generator)
python feature_engineering.py

# Step 2: Cluster traffic states
python clustering.py

# Step 3: Train congestion predictors
python prediction.py

# Step 4: Optimize signal timing
python optimizer.py
```

### 4. Open the Dashboard

Open `dashboard/index.html` in your browser.  
- If the Flask server is running → shows **live sensor data**
- If offline → shows **demo data** automatically

---

## 📊 ML / Optimization Details

| Component            | Method                     | Purpose                              |
|----------------------|----------------------------|--------------------------------------|
| Feature Engineering  | Sliding window (30 samples)| Extract vehicle count, speed, occupancy, flow rate |
| Clustering           | K-Means, DBSCAN           | Identify Free-Flow / Moderate / Congested states   |
| Prediction           | KNN, Decision Tree         | Predict next traffic state from current features    |
| Optimization         | Differential Evolution     | Minimize trajectory variance by tuning green-light splits |

---

## 📈 Output Files (in `data/`)

| File                      | Description                          |
|---------------------------|--------------------------------------|
| `traffic_data.csv`        | Raw / synthetic sensor readings      |
| `features.csv`            | Extracted feature vectors            |
| `features_labeled.csv`    | Features with cluster labels         |
| `kmeans_clusters.png`     | K-Means cluster scatter plots        |
| `dbscan_clusters.png`     | DBSCAN cluster scatter plots         |
| `traffic_timeline.png`    | Traffic state over time              |
| `confusion_matrices.png`  | Prediction accuracy                  |
| `feature_importance.png`  | Decision tree feature importances    |
| `optimization_results.png`| Before/after variance comparison     |
| `optimization_summary.json`| Optimal signal timings             |

---

## 👥 Team

CSE275 Course Project

---

## 📄 License

Academic use only.

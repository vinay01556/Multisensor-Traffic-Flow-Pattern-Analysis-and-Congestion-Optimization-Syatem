"""
IntelliTraffic Pro - Advanced Traffic Intelligence
Backend Server (Flask)

Endpoints:
  POST /api/sensor-data   - Receive JSON sensor readings from ESP32
  GET  /api/sensor-data    - Retrieve stored readings (with optional ?limit=N)
  GET  /api/status         - Health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from database import TrafficDatabase
import datetime
import json
import os
import pickle
import numpy as np

from config import Config
from logger import log
from waitress import serve

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, resources={r"/api/*": {"origins": Config.CORS_ORIGIN}})

db = TrafficDatabase()

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knn_model.pkl")
knn_model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            knn_model = pickle.load(f)
        log.info(f"Loaded KNN model from {MODEL_PATH}")
    except Exception as e:
        log.error(f"Error loading model: {e}", exc_info=True)

@app.route("/api/predict", methods=["GET"])
def predict_congestion():
    """
    IntelliTraffic Pro - Congestion Prediction Module

    Uses a sliding window of recent traffic states to predict
    the upcoming congestion level using KNN / Decision Tree.
    """
    if not knn_model:
        return jsonify({"error": "Model not loaded"}), 503

    # Need at least 30 samples for a proper window (matching feature_engineering.py)
    records = db.get_recent(32)
    if len(records) < 5:
        return jsonify({"error": "Insufficient data for prediction"}), 400

    # Features: vehicle_count, avg_speed, std_speed, avg_distance, occupancy_ratio, flow_rate
    speeds = [r["speed_cm_s"] for r in records]
    dists = [r["distance_cm"] for r in records]
    irs = [r["ir_detected"] for r in records]

    feat = np.array([
        sum(irs),               # vehicle_count
        np.mean(speeds),        # avg_speed
        np.std(speeds) if len(speeds) > 1 else 0, # std_speed
        np.mean(dists),         # avg_distance
        np.mean(irs),           # occupancy_ratio
        np.mean(speeds) * 0.1   # approximate flow_rate
    ]).reshape(1, -1)

    prediction = int(knn_model.predict(feat)[0])
    state_map = {0: "Free-Flow", 1: "Moderate", 2: "Congested"}

    return jsonify({
        "prediction_code": prediction,
        "prediction_name": state_map.get(prediction, "Unknown"),
        "timestamp": datetime.datetime.now().isoformat()
    })


@app.route("/api/predict/manual", methods=["POST"])
def predict_manual():
    """Predict congestion based on manual feature input."""
    if not knn_model:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json(force=True)
        # Required features: vehicle_count, avg_speed, std_speed, avg_distance, occupancy_ratio, flow_rate
        required = ["vehicle_count", "avg_speed", "std_speed", "avg_distance", "occupancy_ratio", "flow_rate"]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        feat = np.array([
            data["vehicle_count"],
            data["avg_speed"],
            data["std_speed"],
            data["avg_distance"],
            data["occupancy_ratio"],
            data["flow_rate"]
        ]).reshape(1, -1)

        prediction = int(knn_model.predict(feat)[0])
        state_map = {0: "Free-Flow", 1: "Moderate", 2: "Congested"}

        return jsonify({
            "prediction_code": prediction,
            "prediction_name": state_map.get(prediction, "Unknown"),
            "timestamp": datetime.datetime.now().isoformat(),
            "input": data
        })

    except Exception as e:
        log.error(f"Error in manual prediction: {e}", exc_info=True)
        return jsonify({"error": "Internal server error during manual prediction"}), 500


@app.route("/api/status", methods=["GET"])
def status():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat(),
        "total_records": db.count()
    })


@app.route("/api/sensor-data", methods=["POST"])
def receive_sensor_data():
    """Receive a JSON sensor reading from ESP32."""
    try:
        data = request.get_json(force=True)

        # Validate required fields
        required = ["ir", "dist_cm", "speed_cm_s", "lane"]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Add server-side timestamp
        data["server_ts"] = datetime.datetime.now().isoformat()

        db.insert(data)

        return jsonify({"status": "stored", "sample": data.get("sample", "?")}), 201

    except Exception as e:
        log.error(f"Error receiving sensor data: {e}", exc_info=True)
        return jsonify({"error": "Internal server error processing sensor data"}), 500


@app.route("/api/sensor-data", methods=["GET"])
def get_sensor_data():
    """Retrieve stored sensor readings."""
    limit = request.args.get("limit", 100, type=int)
    records = db.get_recent(limit)
    return jsonify({"count": len(records), "data": records})


@app.route("/api/sensor-data/all", methods=["GET"])
def get_all_sensor_data():
    """Retrieve ALL stored sensor readings (for analysis)."""
    records = db.get_all()
    return jsonify({"count": len(records), "data": records})


if __name__ == "__main__":
    env = os.getenv("FLASK_ENV", "development")
    
    log.info("=" * 50)
    log.info(" IntelliTraffic Pro - Smart Traffic Intelligence")
    log.info(f" Starting at {datetime.datetime.now().isoformat()} in {env} mode")
    log.info("=" * 50)
    
    if env == "production":
        log.info(f"Serve WSGI using Waitress on {Config.HOST}:{Config.PORT}")
        serve(app, host=Config.HOST, port=Config.PORT)
    else:
        app.run(host=Config.HOST, port=Config.PORT, debug=True)

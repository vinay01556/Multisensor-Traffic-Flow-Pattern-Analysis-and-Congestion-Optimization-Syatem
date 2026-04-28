"""
IntelliTraffic Pro - Advanced Traffic Intelligence
Backend Server (FastAPI)

Endpoints:
  POST /api/sensor-data       - Receive JSON sensor readings from ESP32
  GET  /api/sensor-data       - Retrieve stored readings (with optional ?limit=N)
  GET  /api/sensor-data/all   - Retrieve ALL stored readings
  GET  /api/status            - Health check
  GET  /api/predict           - Predict congestion from recent data
  POST /api/predict/manual    - Predict congestion from manual feature input

Interactive docs available at /docs (Swagger UI) and /redoc (ReDoc).
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from database import TrafficDatabase
import datetime
import os
import pickle
import numpy as np

from config import Config
from logger import log

# ── Pydantic Models ─────────────────────────────────────

class SensorDataIn(BaseModel):
    """Payload sent by the ESP32."""
    ir: int = Field(..., description="IR sensor detection (0 or 1)")
    dist_cm: float = Field(..., description="Ultrasonic distance in cm")
    speed_cm_s: float = Field(..., description="Estimated speed in cm/s")
    lane: int = Field(..., description="Lane identifier")
    sample: Optional[int] = Field(None, description="Sample sequence number")
    ts: Optional[int] = Field(None, description="Device-side timestamp (ms)")


class SensorDataOut(BaseModel):
    status: str
    sample: Any


class ManualPredictIn(BaseModel):
    """Manual feature input for congestion prediction."""
    vehicle_count: float = Field(..., description="Number of vehicles detected")
    avg_speed: float = Field(..., description="Average speed (cm/s)")
    std_speed: float = Field(..., description="Speed standard deviation")
    avg_distance: float = Field(..., description="Average distance (cm)")
    occupancy_ratio: float = Field(..., description="Occupancy ratio (0-1)")
    flow_rate: float = Field(..., description="Traffic flow rate")


class PredictionOut(BaseModel):
    prediction_code: int
    prediction_name: str
    timestamp: str
    input: Optional[dict] = None


class StatusOut(BaseModel):
    status: str
    timestamp: str
    total_records: int


class SensorListOut(BaseModel):
    count: int
    data: List[dict]


# ── App Setup ───────────────────────────────────────────

app = FastAPI(
    title="IntelliTraffic Pro API",
    description="Smart Multisensor Traffic Flow Pattern Analysis & Congestion Optimization",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[Config.CORS_ORIGIN] if Config.CORS_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = TrafficDatabase()

# ── Load ML Model ───────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knn_model.pkl")
knn_model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            knn_model = pickle.load(f)
        log.info(f"Loaded KNN model from {MODEL_PATH}")
    except Exception as e:
        log.error(f"Error loading model: {e}", exc_info=True)


# ── Routes ──────────────────────────────────────────────

@app.get("/api/status", response_model=StatusOut, tags=["Health"])
def status():
    """Health check endpoint."""
    return StatusOut(
        status="ok",
        timestamp=datetime.datetime.now().isoformat(),
        total_records=db.count(),
    )


@app.post("/api/sensor-data", response_model=SensorDataOut, status_code=201, tags=["Sensor Data"])
def receive_sensor_data(payload: SensorDataIn):
    """Receive a JSON sensor reading from ESP32."""
    try:
        data = payload.model_dump()
        data["server_ts"] = datetime.datetime.now().isoformat()
        db.insert(data)
        return SensorDataOut(status="stored", sample=data.get("sample", "?"))
    except Exception as e:
        log.error(f"Error receiving sensor data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing sensor data")


@app.get("/api/sensor-data", response_model=SensorListOut, tags=["Sensor Data"])
def get_sensor_data(limit: int = Query(100, ge=1, le=10000, description="Number of recent records")):
    """Retrieve stored sensor readings (most recent first)."""
    records = db.get_recent(limit)
    return SensorListOut(count=len(records), data=records)


@app.get("/api/sensor-data/all", response_model=SensorListOut, tags=["Sensor Data"])
def get_all_sensor_data():
    """Retrieve ALL stored sensor readings (for analysis)."""
    records = db.get_all()
    return SensorListOut(count=len(records), data=records)


@app.get("/api/predict", response_model=PredictionOut, tags=["Prediction"])
def predict_congestion():
    """
    Congestion Prediction Module

    Uses a sliding window of recent traffic states to predict
    the upcoming congestion level using the KNN model.
    """
    if not knn_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    records = db.get_recent(32)
    if len(records) < 5:
        raise HTTPException(status_code=400, detail="Insufficient data for prediction")

    speeds = [r["speed_cm_s"] for r in records]
    dists = [r["distance_cm"] for r in records]
    irs = [r["ir_detected"] for r in records]

    feat = np.array([
        sum(irs),                                       # vehicle_count
        np.mean(speeds),                                # avg_speed
        np.std(speeds) if len(speeds) > 1 else 0,      # std_speed
        np.mean(dists),                                 # avg_distance
        np.mean(irs),                                   # occupancy_ratio
        np.mean(speeds) * 0.1,                          # approximate flow_rate
    ]).reshape(1, -1)

    prediction = int(knn_model.predict(feat)[0])
    state_map = {0: "Free-Flow", 1: "Moderate", 2: "Congested"}

    return PredictionOut(
        prediction_code=prediction,
        prediction_name=state_map.get(prediction, "Unknown"),
        timestamp=datetime.datetime.now().isoformat(),
    )


@app.post("/api/predict/manual", response_model=PredictionOut, tags=["Prediction"])
def predict_manual(payload: ManualPredictIn):
    """Predict congestion based on manual feature input."""
    if not knn_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        feat = np.array([
            payload.vehicle_count,
            payload.avg_speed,
            payload.std_speed,
            payload.avg_distance,
            payload.occupancy_ratio,
            payload.flow_rate,
        ]).reshape(1, -1)

        prediction = int(knn_model.predict(feat)[0])
        state_map = {0: "Free-Flow", 1: "Moderate", 2: "Congested"}

        return PredictionOut(
            prediction_code=prediction,
            prediction_name=state_map.get(prediction, "Unknown"),
            timestamp=datetime.datetime.now().isoformat(),
            input=payload.model_dump(),
        )

    except Exception as e:
        log.error(f"Error in manual prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during manual prediction")


# ── Entrypoint ──────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    log.info("=" * 50)
    log.info(" IntelliTraffic Pro - Smart Traffic Intelligence")
    log.info(f" Starting at {datetime.datetime.now().isoformat()}")
    log.info("=" * 50)

    uvicorn.run(
        "server:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=True,
    )

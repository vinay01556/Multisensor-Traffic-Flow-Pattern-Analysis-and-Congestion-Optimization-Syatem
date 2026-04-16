"""
CSE275 — Traffic Database Module
Stores sensor readings in SQLite + exports to CSV.
"""

import sqlite3
import csv
import os
import json
import datetime

DB_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DB_DIR, "traffic.db")
CSV_PATH = os.path.join(DB_DIR, "traffic_data.csv")


class TrafficDatabase:
    """Simple SQLite wrapper for sensor data storage."""

    def __init__(self, db_path=DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_table()

    # ── Private ──────────────────────────────────────────

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_table(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample      INTEGER,
                    device_ts   INTEGER,
                    server_ts   TEXT,
                    ir_detected INTEGER,
                    distance_cm REAL,
                    speed_cm_s  REAL,
                    lane_id     INTEGER,
                    raw_json    TEXT
                )
            """)
            conn.commit()

    # ── Public API ───────────────────────────────────────

    def insert(self, data: dict):
        """Insert a single sensor reading."""
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO sensor_readings
                    (sample, device_ts, server_ts, ir_detected, distance_cm, speed_cm_s, lane_id, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("sample"),
                data.get("ts"),
                data.get("server_ts", datetime.datetime.now().isoformat()),
                data.get("ir"),
                data.get("dist_cm"),
                data.get("speed_cm_s"),
                data.get("lane"),
                json.dumps(data)
            ))
            conn.commit()

    def get_recent(self, limit=100) -> list:
        """Get the most recent N readings."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sensor_readings ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all(self) -> list:
        """Get all readings (for analysis pipeline)."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sensor_readings ORDER BY id ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        """Total number of stored readings."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]

    def export_csv(self, path=CSV_PATH):
        """Export all readings to CSV for offline analysis."""
        records = self.get_all()
        if not records:
            print("No records to export.")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        keys = records[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)

        print(f"Exported {len(records)} records to {path}")


if __name__ == "__main__":
    # Quick test
    db = TrafficDatabase()
    import random
    import time

    print("Populating sample data...")
    for i in range(20):
        db.insert({
            "sample": i,
            "ts": int(time.time() * 1000) + i*100,
            "ir": 1 if random.random() > 0.6 else 0,
            "dist_cm": 50 + random.random() * 50,
            "speed_cm_s": 30 + random.random() * 40,
            "lane": 1
        })
    print(f"Total records: {db.count()}")
    print("Recent:", db.get_recent(5))

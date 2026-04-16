/*
 * IntelliTraffic Pro — Multisensor Traffic Flow Analysis
 * ESP32 WiFi Gateway
 *
 * Reads JSON sensor data from Arduino over Serial2,
 * forwards it to the Python backend via HTTP POST.
 *
 * Wiring:
 *   Arduino TX  → ESP32 GPIO 16 (RX2)
 *   Common GND
 */

#include <WiFi.h>
#include <HTTPClient.h>

// ───── WiFi Credentials (update for your network) ─────
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// ───── Backend Server ─────
const char* SERVER_URL = "http://192.168.1.100:5000/api/sensor-data";

// ───── Serial2 for Arduino communication ─────
#define RXD2 16   // ESP32 GPIO 16
#define TXD2 17   // ESP32 GPIO 17 (not used for RX-only)

// ───── LED for status ─────
#define STATUS_LED 2

void setup() {
  Serial.begin(115200);          // Debug console
  Serial2.begin(9600, SERIAL_8N1, RXD2, TXD2);  // Arduino link

  pinMode(STATUS_LED, OUTPUT);
  digitalWrite(STATUS_LED, LOW);

  // ── Connect to WiFi ──
  Serial.println("\n[ESP32] Connecting to WiFi...");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[ESP32] WiFi connected!");
    Serial.print("[ESP32] IP: ");
    Serial.println(WiFi.localIP());
    blinkLED(3, 200);  // 3 quick blinks = success
  } else {
    Serial.println("\n[ESP32] WiFi FAILED — running in offline mode");
    blinkLED(10, 100); // rapid blinks = error
  }
}

void loop() {
  // ── Read a full JSON line from Arduino ──
  if (Serial2.available()) {
    String jsonLine = Serial2.readStringUntil('\n');
    jsonLine.trim();

    if (jsonLine.length() == 0 || jsonLine.charAt(0) != '{') {
      return;  // Skip non-JSON lines
    }

    Serial.print("[RX] ");
    Serial.println(jsonLine);

    // ── Forward to backend ──
    if (WiFi.status() == WL_CONNECTED) {
      sendToServer(jsonLine);
    } else {
      Serial.println("[ESP32] WiFi not connected — data dropped");
      // Attempt reconnect every 30 seconds handled below
      reconnectWiFi();
    }
  }
}

// ───── HTTP POST Helper ─────
void sendToServer(String payload) {
  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(3000);  // 3-second timeout

  int httpCode = http.POST(payload);

  if (httpCode > 0) {
    Serial.printf("[HTTP] POST response: %d\n", httpCode);
    digitalWrite(STATUS_LED, HIGH);
    delay(50);
    digitalWrite(STATUS_LED, LOW);
  } else {
    Serial.printf("[HTTP] POST failed: %s\n", http.errorToString(httpCode).c_str());
  }

  http.end();
}

// ───── WiFi Reconnect ─────
unsigned long lastReconnectAttempt = 0;

void reconnectWiFi() {
  unsigned long now = millis();
  if (now - lastReconnectAttempt > 30000) {  // Every 30 s
    lastReconnectAttempt = now;
    Serial.println("[ESP32] Attempting WiFi reconnect...");
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  }
}

// ───── LED Utility ─────
void blinkLED(int times, int delayMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(STATUS_LED, HIGH);
    delay(delayMs);
    digitalWrite(STATUS_LED, LOW);
    delay(delayMs);
  }
}

/*
 * IntelliTraffic Pro — Multisensor Traffic Flow Analysis
 * Arduino Sensor Sketch
 * 
 * Hardware:
 *   - IR sensor  → Digital pin 2 (vehicle presence detection)
 *   - HC-SR04    → Trig pin 9, Echo pin 10 (distance / speed estimation)
 * 
 * Sends JSON data over Serial to ESP32 at 9600 baud.
 */

// ───── Pin Definitions ─────
#define IR_PIN        2
#define TRIG_PIN      9
#define ECHO_PIN      10
#define ONBOARD_LED   13

// ───── Configuration ─────
#define SAMPLE_INTERVAL_MS  200   // Sampling every 200 ms
#define LANE_ID             1     // Change per lane deployment
#define SPEED_WINDOW        2     // Number of consecutive readings for speed calc

// ───── Global State ─────
float prevDistance = -1;
unsigned long prevTime = 0;
unsigned long sampleCount = 0;

void setup() {
  Serial.begin(9600);

  pinMode(IR_PIN,     INPUT);
  pinMode(TRIG_PIN,   OUTPUT);
  pinMode(ECHO_PIN,   INPUT);
  pinMode(ONBOARD_LED, OUTPUT);

  digitalWrite(TRIG_PIN, LOW);
  delay(100);

  Serial.println("{\"status\":\"sensor_init_ok\"}");
}

void loop() {
  unsigned long now = millis();

  // ── 1. Read IR sensor (vehicle presence) ──
  int irDetected = digitalRead(IR_PIN) == LOW ? 1 : 0;   // LOW = object detected for most IR modules

  // ── 2. Read ultrasonic distance ──
  float distanceCm = readUltrasonic();

  // ── 3. Estimate speed (cm/s) from consecutive distance changes ──
  float speedEstimate = 0.0;
  if (prevDistance >= 0 && prevTime > 0) {
    float dt = (now - prevTime) / 1000.0;  // seconds
    if (dt > 0) {
      speedEstimate = abs(distanceCm - prevDistance) / dt;
    }
  }
  prevDistance = distanceCm;
  prevTime    = now;

  // ── 4. LED feedback ──
  digitalWrite(ONBOARD_LED, irDetected ? HIGH : LOW);

  // ── 5. Send JSON packet over Serial ──
  sampleCount++;
  Serial.print("{\"sample\":");
  Serial.print(sampleCount);
  Serial.print(",\"ts\":");
  Serial.print(now);
  Serial.print(",\"ir\":");
  Serial.print(irDetected);
  Serial.print(",\"dist_cm\":");
  Serial.print(distanceCm, 1);
  Serial.print(",\"speed_cm_s\":");
  Serial.print(speedEstimate, 2);
  Serial.print(",\"lane\":");
  Serial.print(LANE_ID);
  Serial.println("}");

  delay(SAMPLE_INTERVAL_MS);
}

// ───── Ultrasonic Helper ─────
float readUltrasonic() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);  // 30 ms timeout (~5 m range)

  if (duration == 0) return -1.0;  // No echo / out of range

  float distance = (duration * 0.0343) / 2.0;  // speed of sound ≈ 343 m/s
  return distance;
}

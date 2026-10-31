#include <Arduino.h>
#include <math.h>

// ============================================================
// VISENTRA ESP32
// TWO-ZONE PROXIMITY WARNING SYSTEM
// Speed sensor removed. Buzzer retained.
// ============================================================

// SECTION 01 - HARDWARE PINS

#define LEFT_LED_PIN             25
#define RIGHT_LED_PIN            26
#define SWITCH_PIN               14
#define LDR_PIN                  15

#define LEFT_TRIG_PIN             5
#define LEFT_ECHO_PIN            18

#define RIGHT_TRIG_PIN           32
#define RIGHT_ECHO_PIN           33

#define BUZZER_PIN                4

// SECTION 02 - FEATURE SETTINGS

#define ENABLE_LEFT_SENSOR       true
#define ENABLE_RIGHT_SENSOR      true

#define ENABLE_LEFT_LED          true
#define ENABLE_RIGHT_LED         true

#define ENABLE_BUZZER            true
#define ENABLE_LDR               true
#define ENABLE_DAY_NIGHT         true
#define ENABLE_MODE_SWITCH       true

#define ENABLE_CRITICAL_ALERT    true
#define ENABLE_MULTIPLE_ALERT    true
#define ENABLE_SENSOR_TIMEOUT    true

#define ENABLE_SERIAL_MONITOR    true
#define ENABLE_SERIAL_DEBUG      true

// SECTION 03 - DISTANCE SETTINGS (mm)

#define CRITICAL_DISTANCE_MM     500.0f
#define WARNING_DISTANCE_MM      1500.0f
#define SAFE_DISTANCE_MM         3000.0f
#define MAX_DISTANCE_MM          4000.0f

#define LEFT_CRITICAL_MM         500.0f
#define RIGHT_CRITICAL_MM        500.0f

#define LEFT_WARNING_MM          1500.0f
#define RIGHT_WARNING_MM         1500.0f

#define DISTANCE_HYSTERESIS_MM   150.0f
#define MIN_VALID_DISTANCE_MM    30.0f

#define ULTRASONIC_TIMEOUT_US    25000UL
#define SENSOR_SAMPLE_INTERVAL   70UL
#define DISTANCE_FILTER_SAMPLES  3

#define INVALID_DISTANCE_MM      99999.0f

// SECTION 04 - OPERATING MODES

enum OperatingMode {
  PARKING_MODE = 0,
  DAILY_ROAD_MODE,
  TRAFFIC_MODE,
  LONG_DRIVE_MODE
};

OperatingMode currentMode = PARKING_MODE;

#define TOTAL_MODES              4
#define MODE_DEBOUNCE_MS         250UL
#define MODE_CHANGE_BEEP_MS      100UL

// SECTION 05 - LED SETTINGS

#define LED_ACTIVE_HIGH          true

#define NORMAL_BLINK_ON_MS       250UL
#define NORMAL_BLINK_OFF_MS      250UL

#define WARNING_BLINK_ON_MS      150UL
#define WARNING_BLINK_OFF_MS     150UL

#define CRITICAL_BLINK_ON_MS      70UL
#define CRITICAL_BLINK_OFF_MS     70UL

#define MULTIPLE_BLINK_ON_MS     100UL
#define MULTIPLE_BLINK_OFF_MS    100UL

#define DAY_LED_ON_TIME_MS       150UL
#define DAY_LED_OFF_TIME_MS      350UL

#define NIGHT_LED_ON_TIME_MS     300UL
#define NIGHT_LED_OFF_TIME_MS    200UL

// SECTION 06 - BUZZER SETTINGS

#define BUZZER_ACTIVE_HIGH       true
#define BUZZER_SILENT            false

#define BUZZER_WARNING_ON_MS     100UL
#define BUZZER_WARNING_OFF_MS    900UL

#define BUZZER_APPROACH_ON_MS    200UL
#define BUZZER_APPROACH_OFF_MS   300UL

#define BUZZER_CRITICAL_ON_MS    500UL
#define BUZZER_CRITICAL_OFF_MS   100UL

#define BUZZER_MULTIPLE_ON_MS    150UL
#define BUZZER_MULTIPLE_OFF_MS   150UL

// SECTION 07 - LDR SETTINGS

#define LDR_DARK_THRESHOLD       1500
#define LDR_BRIGHT_THRESHOLD     2500

#define LDR_SAMPLE_INTERVAL_MS   500UL
#define LDR_FILTER_ALPHA         0.20f

#define LDR_ADC_MAX              4095

// SECTION 08 - SYSTEM TIMING

#define SENSOR_RECOVERY_MS       250UL
#define SENSOR_STALE_MS          1000UL

#define SYSTEM_STARTUP_DELAY_MS  500UL
#define SERIAL_INTERVAL_MS       1000UL

#define WARNING_COOLDOWN_MS      500UL
#define CRITICAL_COOLDOWN_MS     100UL
#define MULTIPLE_COOLDOWN_MS     500UL

// SECTION 09 - ADVANCED SETTINGS

#define ENABLE_DISTANCE_FILTER   true
#define ENABLE_DISTANCE_AVERAGE  true
#define ENABLE_MODE_THRESHOLDS   true
#define ENABLE_ALERT_PRIORITY    true
#define ENABLE_SENSOR_RECOVERY   true
#define ENABLE_AUTO_RESET        true
#define ENABLE_STARTUP_TEST      true

// SECTION 10 - ULTRASONIC SCHEDULING

#define ULTRASONIC_TRIGGER_GAP_MS 70UL

// ============================================================
// SECTION 11 - ALERT TYPES
// ============================================================

enum AlertType {
  ALERT_NONE = 0,
  ALERT_LEFT,
  ALERT_RIGHT,
  ALERT_MULTIPLE,
  ALERT_CRITICAL
};

AlertType currentAlert = ALERT_NONE;

// ============================================================
// SECTION 12 - MODE DISTANCE MULTIPLIERS
// ============================================================

float modeDistanceMultiplier() {
  if (!ENABLE_MODE_THRESHOLDS) {
    return 1.0f;
  }

  switch (currentMode) {
    case PARKING_MODE:
      return 0.75f;

    case DAILY_ROAD_MODE:
      return 1.00f;

    case TRAFFIC_MODE:
      return 0.80f;

    case LONG_DRIVE_MODE:
      return 1.20f;

    default:
      return 1.00f;
  }
}

// ============================================================
// SECTION 13 - DISTANCE VARIABLES
// ============================================================

float leftDistance = INVALID_DISTANCE_MM;
float rightDistance = INVALID_DISTANCE_MM;

float filteredLeftDistance = INVALID_DISTANCE_MM;
float filteredRightDistance = INVALID_DISTANCE_MM;

float leftSamples[DISTANCE_FILTER_SAMPLES] = {0};
float rightSamples[DISTANCE_FILTER_SAMPLES] = {0};

uint8_t leftSampleCount = 0;
uint8_t rightSampleCount = 0;

uint8_t leftSampleIndex = 0;
uint8_t rightSampleIndex = 0;

bool leftSensorValid = false;
bool rightSensorValid = false;

bool leftSensorTimedOut = false;
bool rightSensorTimedOut = false;

unsigned long lastLeftValidTime = 0;
unsigned long lastRightValidTime = 0;

unsigned long leftRecoveryUntil = 0;
unsigned long rightRecoveryUntil = 0;

// ============================================================
// SECTION 14 - DETECTION VARIABLES
// ============================================================

bool leftDetected = false;
bool rightDetected = false;

bool leftCritical = false;
bool rightCritical = false;

bool multipleDetected = false;

// ============================================================
// SECTION 15 - LDR VARIABLES
// ============================================================

float filteredLDR = 0.0f;

bool ldrInitialized = false;
bool isNight = false;

unsigned long lastLDRUpdate = 0;

// ============================================================
// SECTION 16 - SYSTEM TIMERS
// ============================================================

unsigned long lastSerialUpdate = 0;
unsigned long lastSensorTrigger = 0;

unsigned long lastLEDUpdate = 0;
unsigned long lastBuzzerUpdate = 0;

unsigned long lastModePress = 0;
unsigned long lastAlertLog = 0;

unsigned long modeBeepUntil = 0;

bool ledState = false;
bool buzzerState = false;

// ============================================================
// SECTION 17 - ULTRASONIC STATE MACHINE
// ============================================================

enum SensorSide {
  SENSOR_NONE = 0,
  SENSOR_LEFT,
  SENSOR_RIGHT
};

enum UltrasonicState {
  ULTRA_IDLE = 0,
  ULTRA_WAIT_RISE,
  ULTRA_WAIT_FALL
};

SensorSide activeSensor = SENSOR_NONE;
SensorSide nextSensor = SENSOR_LEFT;

UltrasonicState ultrasonicState = ULTRA_IDLE;

unsigned long ultrasonicTriggerTime = 0;
unsigned long echoRiseTime = 0;

// ============================================================
// SECTION 18 - OUTPUT HELPERS
// ============================================================

void setLeftLED(bool state) {
  if (!ENABLE_LEFT_LED) {
    digitalWrite(LEFT_LED_PIN, LOW);
    return;
  }

  digitalWrite(
    LEFT_LED_PIN,
    state == LED_ACTIVE_HIGH ? HIGH : LOW
  );
}

void setRightLED(bool state) {
  if (!ENABLE_RIGHT_LED) {
    digitalWrite(RIGHT_LED_PIN, LOW);
    return;
  }

  digitalWrite(
    RIGHT_LED_PIN,
    state == LED_ACTIVE_HIGH ? HIGH : LOW
  );
}

void setBuzzer(bool state) {
  if (!ENABLE_BUZZER || BUZZER_SILENT) {
    digitalWrite(BUZZER_PIN, LOW);
    return;
  }

  digitalWrite(
    BUZZER_PIN,
    state == BUZZER_ACTIVE_HIGH ? HIGH : LOW
  );
}

// ============================================================
// SECTION 19 - DISTANCE MEDIAN FILTER
// ============================================================

float medianOfThree(float a, float b, float c) {
  if (a > b) {
    float temp = a;
    a = b;
    b = temp;
  }

  if (b > c) {
    float temp = b;
    b = c;
    c = temp;
  }

  if (a > b) {
    float temp = a;
    a = b;
    b = temp;
  }

  return b;
}

float filterDistance(
  float reading,
  float *samples,
  uint8_t &sampleCount,
  uint8_t &sampleIndex,
  float previous
) {
  if (reading >= INVALID_DISTANCE_MM) {
    return previous;
  }

  if (!ENABLE_DISTANCE_FILTER) {
    return reading;
  }

  samples[sampleIndex] = reading;

  sampleIndex =
    (sampleIndex + 1) % DISTANCE_FILTER_SAMPLES;

  if (sampleCount < DISTANCE_FILTER_SAMPLES) {
    sampleCount++;
  }

  float result = reading;

  if (ENABLE_DISTANCE_AVERAGE &&
      sampleCount == DISTANCE_FILTER_SAMPLES) {
    result = medianOfThree(
      samples[0],
      samples[1],
      samples[2]
    );
  }

  if (previous >= INVALID_DISTANCE_MM) {
    return result;
  }

  return previous * 0.65f + result * 0.35f;
}

// ============================================================
// SECTION 20 - PROCESS SENSOR READING
// ============================================================

void processDistanceReading(
  SensorSide side,
  unsigned long durationUs
) {
  float distance = (durationUs * 0.343f) / 2.0f;

  bool valid =
    distance >= MIN_VALID_DISTANCE_MM &&
    distance <= MAX_DISTANCE_MM;

  unsigned long now = millis();

  if (side == SENSOR_LEFT) {
    if (valid) {
      leftDistance = distance;

      filteredLeftDistance = filterDistance(
        distance,
        leftSamples,
        leftSampleCount,
        leftSampleIndex,
        filteredLeftDistance
      );

      leftSensorValid = true;
      leftSensorTimedOut = false;
      lastLeftValidTime = now;
    } else {
      leftDistance = INVALID_DISTANCE_MM;
      leftSensorValid = false;
    }
  }

  if (side == SENSOR_RIGHT) {
    if (valid) {
      rightDistance = distance;

      filteredRightDistance = filterDistance(
        distance,
        rightSamples,
        rightSampleCount,
        rightSampleIndex,
        filteredRightDistance
      );

      rightSensorValid = true;
      rightSensorTimedOut = false;
      lastRightValidTime = now;
    } else {
      rightDistance = INVALID_DISTANCE_MM;
      rightSensorValid = false;
    }
  }
}

// ============================================================
// SECTION 21 - ULTRASONIC SENSOR FAILURE
// ============================================================

void handleSensorTimeout(SensorSide side) {
  unsigned long now = millis();

  if (side == SENSOR_LEFT) {
    leftDistance = INVALID_DISTANCE_MM;
    leftSensorValid = false;
    leftSensorTimedOut = true;

    if (ENABLE_SENSOR_RECOVERY) {
      leftRecoveryUntil = now + SENSOR_RECOVERY_MS;
    }
  }

  if (side == SENSOR_RIGHT) {
    rightDistance = INVALID_DISTANCE_MM;
    rightSensorValid = false;
    rightSensorTimedOut = true;

    if (ENABLE_SENSOR_RECOVERY) {
      rightRecoveryUntil = now + SENSOR_RECOVERY_MS;
    }
  }
}

// ============================================================
// SECTION 22 - START ULTRASONIC SENSOR
// ============================================================

void startUltrasonic(SensorSide side) {
  if (ultrasonicState != ULTRA_IDLE) {
    return;
  }

  unsigned long now = millis();

  if (now - lastSensorTrigger <
      ULTRASONIC_TRIGGER_GAP_MS) {
    return;
  }

  if (side == SENSOR_LEFT) {
    if (!ENABLE_LEFT_SENSOR) {
      return;
    }

    if (ENABLE_SENSOR_RECOVERY &&
        (int32_t)(now - leftRecoveryUntil) < 0) {
      return;
    }

    digitalWrite(LEFT_TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(LEFT_TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(LEFT_TRIG_PIN, LOW);
  }

  if (side == SENSOR_RIGHT) {
    if (!ENABLE_RIGHT_SENSOR) {
      return;
    }

    if (ENABLE_SENSOR_RECOVERY &&
        (int32_t)(now - rightRecoveryUntil) < 0) {
      return;
    }

    digitalWrite(RIGHT_TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(RIGHT_TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(RIGHT_TRIG_PIN, LOW);
  }

  activeSensor = side;
  ultrasonicState = ULTRA_WAIT_RISE;

  ultrasonicTriggerTime = micros();
  lastSensorTrigger = now;
}

// ============================================================
// SECTION 23 - NON-BLOCKING ULTRASONIC UPDATE
// ============================================================

void updateDistanceSensors() {
  if (ultrasonicState == ULTRA_IDLE) {
    if (nextSensor == SENSOR_LEFT) {
      startUltrasonic(SENSOR_LEFT);
      nextSensor = SENSOR_RIGHT;
    } else {
      startUltrasonic(SENSOR_RIGHT);
      nextSensor = SENSOR_LEFT;
    }

    return;
  }

  int echoPin =
    activeSensor == SENSOR_LEFT
    ? LEFT_ECHO_PIN
    : RIGHT_ECHO_PIN;

  unsigned long nowUs = micros();

  if (ultrasonicState == ULTRA_WAIT_RISE) {
    if (digitalRead(echoPin) == HIGH) {
      echoRiseTime = nowUs;
      ultrasonicState = ULTRA_WAIT_FALL;
    }
  } else if (ultrasonicState == ULTRA_WAIT_FALL) {
    if (digitalRead(echoPin) == LOW) {
      unsigned long duration = nowUs - echoRiseTime;

      processDistanceReading(activeSensor, duration);

      ultrasonicState = ULTRA_IDLE;
      activeSensor = SENSOR_NONE;
      return;
    }
  }

  if ((unsigned long)(nowUs - ultrasonicTriggerTime) >
      ULTRASONIC_TIMEOUT_US) {
    SensorSide failedSensor = activeSensor;

    if (ENABLE_SENSOR_TIMEOUT) {
      handleSensorTimeout(failedSensor);
    }

    ultrasonicState = ULTRA_IDLE;
    activeSensor = SENSOR_NONE;
  }
}

// ============================================================
// SECTION 24 - SENSOR HEALTH AND STALE DATA
// ============================================================

void updateSensorHealth() {
  if (!ENABLE_AUTO_RESET) {
    return;
  }

  unsigned long now = millis();

  if (ENABLE_LEFT_SENSOR &&
      leftSensorValid &&
      now - lastLeftValidTime > SENSOR_STALE_MS) {
    leftSensorValid = false;
    leftDistance = INVALID_DISTANCE_MM;
    filteredLeftDistance = INVALID_DISTANCE_MM;
    leftSampleCount = 0;
    leftSampleIndex = 0;
    leftDetected = false;
    leftCritical = false;
  }

  if (ENABLE_RIGHT_SENSOR &&
      rightSensorValid &&
      now - lastRightValidTime > SENSOR_STALE_MS) {
    rightSensorValid = false;
    rightDistance = INVALID_DISTANCE_MM;
    filteredRightDistance = INVALID_DISTANCE_MM;
    rightSampleCount = 0;
    rightSampleIndex = 0;
    rightDetected = false;
    rightCritical = false;
  }

  if (!ENABLE_SENSOR_TIMEOUT) {
    leftSensorTimedOut = false;
    rightSensorTimedOut = false;
  }
}

// ============================================================
// SECTION 25 - LDR DAY / NIGHT DETECTION
// ============================================================

void updateLDR() {
  if (!ENABLE_LDR || !ENABLE_DAY_NIGHT) {
    isNight = false;
    return;
  }

  unsigned long now = millis();

  if (now - lastLDRUpdate < LDR_SAMPLE_INTERVAL_MS) {
    return;
  }

  lastLDRUpdate = now;

  int reading = analogRead(LDR_PIN);

  if (!ldrInitialized) {
    filteredLDR = reading;
    ldrInitialized = true;
  } else {
    filteredLDR =
      filteredLDR * (1.0f - LDR_FILTER_ALPHA) +
      reading * LDR_FILTER_ALPHA;
  }

  if (filteredLDR < LDR_DARK_THRESHOLD) {
    isNight = true;
  } else if (filteredLDR > LDR_BRIGHT_THRESHOLD) {
    isNight = false;
  }
}

// ============================================================
// SECTION 26 - DISTANCE HYSTERESIS
// ============================================================

bool updateHysteresis(
  bool previousState,
  float distance,
  float threshold,
  bool sensorValid
) {
  if (!sensorValid ||
      distance >= INVALID_DISTANCE_MM) {
    return false;
  }

  if (previousState) {
    return distance <
      threshold + DISTANCE_HYSTERESIS_MM;
  }

  return distance < threshold;
}

// ============================================================
// SECTION 27 - LEFT / RIGHT DETECTION
// ============================================================

void updateDetection() {
  float multiplier = modeDistanceMultiplier();

  float leftWarning =
    LEFT_WARNING_MM * multiplier;

  float rightWarning =
    RIGHT_WARNING_MM * multiplier;

  float leftCriticalThreshold =
    LEFT_CRITICAL_MM * multiplier;

  float rightCriticalThreshold =
    RIGHT_CRITICAL_MM * multiplier;

  if (ENABLE_LEFT_SENSOR) {
    leftDetected = updateHysteresis(
      leftDetected,
      filteredLeftDistance,
      leftWarning,
      leftSensorValid
    );

    leftCritical = updateHysteresis(
      leftCritical,
      filteredLeftDistance,
      leftCriticalThreshold,
      leftSensorValid
    );
  } else {
    leftDetected = false;
    leftCritical = false;
  }

  if (ENABLE_RIGHT_SENSOR) {
    rightDetected = updateHysteresis(
      rightDetected,
      filteredRightDistance,
      rightWarning,
      rightSensorValid
    );

    rightCritical = updateHysteresis(
      rightCritical,
      filteredRightDistance,
      rightCriticalThreshold,
      rightSensorValid
    );
  } else {
    rightDetected = false;
    rightCritical = false;
  }

  multipleDetected =
    ENABLE_MULTIPLE_ALERT &&
    leftDetected &&
    rightDetected;
}

// ============================================================
// SECTION 28 - ALERT DECISION ENGINE
// ============================================================

void updateAlert() {
  AlertType newAlert = ALERT_NONE;

  if (ENABLE_ALERT_PRIORITY) {
    if (ENABLE_CRITICAL_ALERT &&
        (leftCritical || rightCritical)) {
      newAlert = ALERT_CRITICAL;
    } else if (multipleDetected) {
      newAlert = ALERT_MULTIPLE;
    } else if (leftDetected && rightDetected) {
      newAlert = ALERT_MULTIPLE;
    } else if (leftDetected) {
      newAlert = ALERT_LEFT;
    } else if (rightDetected) {
      newAlert = ALERT_RIGHT;
    }
  } else {
    if (leftDetected && rightDetected) {
      newAlert = ALERT_MULTIPLE;
    } else if (leftDetected) {
      newAlert = ALERT_LEFT;
    } else if (rightDetected) {
      newAlert = ALERT_RIGHT;
    } else if (ENABLE_CRITICAL_ALERT &&
               (leftCritical || rightCritical)) {
      newAlert = ALERT_CRITICAL;
    }
  }

  if (newAlert != currentAlert) {
    currentAlert = newAlert;

    ledState = true;
    buzzerState = true;

    lastLEDUpdate = millis();
    lastBuzzerUpdate = millis();

    if (ENABLE_SERIAL_DEBUG) {
      unsigned long cooldown = WARNING_COOLDOWN_MS;

      if (currentAlert == ALERT_CRITICAL) {
        cooldown = CRITICAL_COOLDOWN_MS;
      } else if (currentAlert == ALERT_MULTIPLE) {
        cooldown = MULTIPLE_COOLDOWN_MS;
      }

      if (millis() - lastAlertLog >= cooldown) {
        Serial.print("VISENTRA ALERT: ");
        Serial.println((int)currentAlert);
        lastAlertLog = millis();
      }
    }
  }
}

// ============================================================
// SECTION 29 - LED PATTERN CONTROL
// ============================================================

void updateLEDs() {
  if (currentAlert == ALERT_NONE) {
    ledState = false;
    setLeftLED(false);
    setRightLED(false);
    return;
  }

  unsigned long onTime = NORMAL_BLINK_ON_MS;
  unsigned long offTime = NORMAL_BLINK_OFF_MS;

  switch (currentAlert) {
    case ALERT_CRITICAL:
      onTime = CRITICAL_BLINK_ON_MS;
      offTime = CRITICAL_BLINK_OFF_MS;
      break;

    case ALERT_MULTIPLE:
      onTime = MULTIPLE_BLINK_ON_MS;
      offTime = MULTIPLE_BLINK_OFF_MS;
      break;

    case ALERT_LEFT:
    case ALERT_RIGHT:
      onTime = WARNING_BLINK_ON_MS;
      offTime = WARNING_BLINK_OFF_MS;
      break;

    default:
      break;
  }

  if (isNight) {
    onTime = NIGHT_LED_ON_TIME_MS;
    offTime = NIGHT_LED_OFF_TIME_MS;
  } else {
    onTime = DAY_LED_ON_TIME_MS;
    offTime = DAY_LED_OFF_TIME_MS;
  }

  unsigned long interval = ledState ? onTime : offTime;

  if (millis() - lastLEDUpdate >= interval) {
    lastLEDUpdate = millis();
    ledState = !ledState;
  }

  bool leftOutput = false;
  bool rightOutput = false;

  switch (currentAlert) {
    case ALERT_CRITICAL:
    case ALERT_MULTIPLE:
      leftOutput = ledState;
      rightOutput = ledState;
      break;

    case ALERT_LEFT:
      leftOutput = ledState;
      break;

    case ALERT_RIGHT:
      rightOutput = ledState;
      break;

    default:
      break;
  }

  setLeftLED(leftOutput);
  setRightLED(rightOutput);
}

// ============================================================
// SECTION 30 - BUZZER PATTERN CONTROL
// ============================================================

void updateBuzzer() {
  if (!ENABLE_BUZZER || BUZZER_SILENT) {
    buzzerState = false;
    setBuzzer(false);
    return;
  }

  // Short beep after changing operating mode.
  if ((int32_t)(modeBeepUntil - millis()) > 0) {
    setBuzzer(true);
    return;
  }

  if (currentAlert == ALERT_NONE) {
    buzzerState = false;
    setBuzzer(false);
    return;
  }

  unsigned long onTime = BUZZER_WARNING_ON_MS;
  unsigned long offTime = BUZZER_WARNING_OFF_MS;

  switch (currentAlert) {
    case ALERT_CRITICAL:
      onTime = BUZZER_CRITICAL_ON_MS;
      offTime = BUZZER_CRITICAL_OFF_MS;
      break;

    case ALERT_MULTIPLE:
      onTime = BUZZER_MULTIPLE_ON_MS;
      offTime = BUZZER_MULTIPLE_OFF_MS;
      break;

    case ALERT_LEFT:
    case ALERT_RIGHT:
      onTime = BUZZER_WARNING_ON_MS;
      offTime = BUZZER_WARNING_OFF_MS;
      break;

    default:
      break;
  }

  unsigned long interval = buzzerState ? onTime : offTime;

  if (millis() - lastBuzzerUpdate >= interval) {
    lastBuzzerUpdate = millis();
    buzzerState = !buzzerState;
  }

  setBuzzer(buzzerState);
}

// ============================================================
// SECTION 31 - MODE SWITCH
// ============================================================

void updateModeSwitch() {
  if (!ENABLE_MODE_SWITCH) {
    return;
  }

  static bool stableButtonState = HIGH;
  static bool lastRawButtonState = HIGH;
  static unsigned long lastDebounceTime = 0;

  bool rawState = digitalRead(SWITCH_PIN);

  if (rawState != lastRawButtonState) {
    lastDebounceTime = millis();
    lastRawButtonState = rawState;
  }

  if (millis() - lastDebounceTime >= MODE_DEBOUNCE_MS) {
    if (rawState != stableButtonState) {
      stableButtonState = rawState;

      if (stableButtonState == LOW) {
        currentMode = (OperatingMode)(
          ((int)currentMode + 1) % TOTAL_MODES
        );

        Serial.print("VISENTRA MODE: ");

        switch (currentMode) {
          case PARKING_MODE:
            Serial.println("PARKING");
            break;

          case DAILY_ROAD_MODE:
            Serial.println("DAILY ROAD");
            break;

          case TRAFFIC_MODE:
            Serial.println("TRAFFIC");
            break;

          case LONG_DRIVE_MODE:
            Serial.println("LONG DRIVE");
            break;
        }

        if (ENABLE_BUZZER && !BUZZER_SILENT) {
          modeBeepUntil = millis() + MODE_CHANGE_BEEP_MS;
          setBuzzer(true);
        }
      }
    }
  }
}

// ============================================================
// SECTION 32 - SERIAL DISTANCE DISPLAY
// ============================================================

void printDistance(const char *label, float distance) {
  Serial.print(label);

  if (distance >= INVALID_DISTANCE_MM) {
    Serial.println("NO VALID READING");
  } else {
    Serial.print(distance, 1);
    Serial.println(" mm");
  }
}

// ============================================================
// SECTION 33 - SERIAL STATUS
// ============================================================

void printSystemStatus() {
  if (!ENABLE_SERIAL_MONITOR) {
    return;
  }

  if (millis() - lastSerialUpdate < SERIAL_INTERVAL_MS) {
    return;
  }

  lastSerialUpdate = millis();

  Serial.println();
  Serial.println("========== VISENTRA STATUS ==========");

  printDistance("Left distance: ", leftDistance);
  printDistance("Right distance: ", rightDistance);

  Serial.print("Left sensor: ");
  Serial.println(leftSensorValid ? "VALID" : "INVALID");

  Serial.print("Right sensor: ");
  Serial.println(rightSensorValid ? "VALID" : "INVALID");

  Serial.print("Left timeout: ");
  Serial.println(leftSensorTimedOut ? "YES" : "NO");

  Serial.print("Right timeout: ");
  Serial.println(rightSensorTimedOut ? "YES" : "NO");

  Serial.print("Left detection: ");
  Serial.println(leftDetected ? "YES" : "NO");

  Serial.print("Right detection: ");
  Serial.println(rightDetected ? "YES" : "NO");

  Serial.print("Multiple objects: ");
  Serial.println(multipleDetected ? "YES" : "NO");

  Serial.print("Critical alert: ");
  Serial.println(
    (leftCritical || rightCritical) ? "YES" : "NO"
  );

  Serial.print("Environment: ");
  Serial.println(isNight ? "NIGHT" : "DAY");

  Serial.print("LDR reading: ");
  Serial.println((int)filteredLDR);

  Serial.print("Mode: ");

  switch (currentMode) {
    case PARKING_MODE:
      Serial.println("PARKING");
      break;

    case DAILY_ROAD_MODE:
      Serial.println("DAILY ROAD");
      break;

    case TRAFFIC_MODE:
      Serial.println("TRAFFIC");
      break;

    case LONG_DRIVE_MODE:
      Serial.println("LONG DRIVE");
      break;
  }

  Serial.print("Alert code: ");
  Serial.println((int)currentAlert);

  Serial.println("Speed sensor: REMOVED");
  Serial.println("Buzzer: ENABLED");
  Serial.println("=====================================");
}

// ============================================================
// SECTION 34 - STARTUP TEST
// ============================================================

void startupTest() {
  Serial.println();
  Serial.println("==================================");
  Serial.println("       VISENTRA ESP32 START");
  Serial.println("==================================");

  if (!ENABLE_STARTUP_TEST) {
    return;
  }

  setLeftLED(true);
  delay(200);
  setLeftLED(false);

  setRightLED(true);
  delay(200);
  setRightLED(false);

  // Buzzer startup confirmation.
  setBuzzer(true);
  delay(150);
  setBuzzer(false);

  Serial.println("LED and buzzer test completed.");
}

// ============================================================
// SECTION 35 - SETUP
// ============================================================

void setup() {
  Serial.begin(115200);

  pinMode(LEFT_LED_PIN, OUTPUT);
  pinMode(RIGHT_LED_PIN, OUTPUT);

  pinMode(SWITCH_PIN, INPUT_PULLUP);

  pinMode(LEFT_TRIG_PIN, OUTPUT);
  pinMode(LEFT_ECHO_PIN, INPUT);

  pinMode(RIGHT_TRIG_PIN, OUTPUT);
  pinMode(RIGHT_ECHO_PIN, INPUT);

  pinMode(BUZZER_PIN, OUTPUT);

  pinMode(LDR_PIN, INPUT);
  analogReadResolution(12);

  digitalWrite(LEFT_TRIG_PIN, LOW);
  digitalWrite(RIGHT_TRIG_PIN, LOW);

  setLeftLED(false);
  setRightLED(false);
  setBuzzer(false);

  delay(SYSTEM_STARTUP_DELAY_MS);

  startupTest();

  lastSensorTrigger =
    millis() - ULTRASONIC_TRIGGER_GAP_MS;

  Serial.println("VISENTRA SYSTEM READY");
  Serial.println("Two-zone proximity system active.");
}

// ============================================================
// SECTION 36 - MAIN LOOP
// ============================================================

void loop() {
  updateModeSwitch();

  updateDistanceSensors();
  updateSensorHealth();

  updateLDR();

  updateDetection();
  updateAlert();

  updateLEDs();
  updateBuzzer();

  printSystemStatus();
}

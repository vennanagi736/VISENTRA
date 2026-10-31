/*
  VISENTRA_USB_Bridge.ino
  Initial USB-serial bridge firmware

  Connection:
    Vivo V27 (USB OTG host) -> USB data cable -> ESP32-WROOM-32 board
    Android app sends newline-terminated ASCII commands over USB serial.

  This ESP32-WROOM-32 style board commonly uses a CP2102 USB-to-UART bridge.
  The phone must support USB OTG host mode and the Android app must support
  the USB-UART bridge chip fitted to the board.

  Existing VISENTRA output pins:
    Left LED  : GPIO 25
    Right LED : GPIO 26
    Buzzer    : GPIO 4

  Serial settings: 115200 baud, 8 data bits, no parity, 1 stop bit.

  Commands (each must end with newline):
    LEFT\n    RIGHT\n
    CENTER\n    BOTH\n    SAFE\n    STOP\n    PING\n    STATUS\n
  Replies:
    ACK,LEFT / ACK,RIGHT / ACK,CENTER / ACK,BOTH / ACK,SAFE
    PONG
    STATUS,<state>
    ERR,<reason>

  Important:
    - This is an initial USB communication test firmware, not the complete
      VISENTRA sensor/detection application.
    - It uses GPIO 25, 26 and 4 only. It does not read the ultrasonic sensors
      or LDR, and it does not change the existing sensor logic.
    - GPIO4 output assumes an active buzzer module suitable for 3.3 V logic.
      Do not drive a high-current buzzer directly from an ESP32 GPIO.
    - Use suitable LED resistors or driver circuitry.
    - A charge-only cable will not work; use a data-capable OTG connection.
*/

#include <Arduino.h>
#include <string.h>

// -------------------- VISENTRA OUTPUT PINS --------------------
constexpr uint8_t LEFT_LED_PIN  = 25;
constexpr uint8_t RIGHT_LED_PIN = 26;
constexpr uint8_t BUZZER_PIN    = 4;

// Change these levels if your LED/buzzer driver is active-low.
constexpr uint8_t LED_ON_LEVEL      = HIGH;
constexpr uint8_t LED_OFF_LEVEL     = LOW;
constexpr uint8_t BUZZER_ON_LEVEL   = HIGH;
constexpr uint8_t BUZZER_OFF_LEVEL  = LOW;

// -------------------- USB SERIAL PROTOCOL --------------------
constexpr uint32_t SERIAL_BAUD = 115200;
constexpr size_t COMMAND_BUFFER_SIZE = 48;
char commandBuffer[COMMAND_BUFFER_SIZE];
size_t commandLength = 0;

// -------------------- WARNING STATES --------------------
enum WarningState : uint8_t {
  STATE_SAFE,
  STATE_LEFT,
  STATE_RIGHT,
  STATE_CENTER,
  STATE_BOTH
};

WarningState currentState = STATE_SAFE;
bool patternOn = false;
uint32_t lastPatternChangeMs = 0;

constexpr uint32_t SIDE_ON_MS   = 180;
constexpr uint32_t SIDE_OFF_MS  = 180;
constexpr uint32_t CENTER_ON_MS = 100;
constexpr uint32_t CENTER_OFF_MS = 100;
constexpr uint32_t BOTH_ON_MS   = 120;
constexpr uint32_t BOTH_OFF_MS  = 120;

void writeOutputs(bool leftOn, bool rightOn, bool buzzerOn) {
  digitalWrite(LEFT_LED_PIN, leftOn ? LED_ON_LEVEL : LED_OFF_LEVEL);
  digitalWrite(RIGHT_LED_PIN, rightOn ? LED_ON_LEVEL : LED_OFF_LEVEL);
  digitalWrite(BUZZER_PIN, buzzerOn ? BUZZER_ON_LEVEL : BUZZER_OFF_LEVEL);
}

const char* stateToText(WarningState state) {
  switch (state) {
    case STATE_LEFT:   return "LEFT";
    case STATE_RIGHT:  return "RIGHT";
    case STATE_CENTER: return "CENTER";
    case STATE_BOTH:   return "BOTH";
    case STATE_SAFE:
    default:           return "SAFE";
  }
}

void sendStatus() {
  Serial.print("STATUS,");
  Serial.println(stateToText(currentState));
}

void setWarningState(WarningState nextState) {
  currentState = nextState;
  patternOn = (nextState != STATE_SAFE);
  lastPatternChangeMs = millis();

  if (nextState == STATE_SAFE) {
    writeOutputs(false, false, false);
    return;
  }

  // Give immediate visual/audio feedback when a warning command arrives.
  const bool leftOn = (nextState == STATE_LEFT ||
                       nextState == STATE_CENTER ||
                       nextState == STATE_BOTH);
  const bool rightOn = (nextState == STATE_RIGHT ||
                        nextState == STATE_CENTER ||
                        nextState == STATE_BOTH);
  writeOutputs(leftOn, rightOn, true);
}

void processCommand(char* command) {
  // Trim leading/trailing spaces and tabs.
  while (*command == ' ' || *command == '\t') {
    ++command;
  }

  size_t length = strlen(command);
  while (length > 0 &&
         (command[length - 1] == ' ' || command[length - 1] == '\t')) {
    command[--length] = '\0';
  }

  // Commands are case-insensitive.
  for (size_t i = 0; command[i] != '\0'; ++i) {
    if (command[i] >= 'a' && command[i] <= 'z') {
      command[i] = static_cast<char>(command[i] - 'a' + 'A');
    }
  }

  if (strcmp(command, "LEFT") == 0) {
    setWarningState(STATE_LEFT);
    Serial.println("ACK,LEFT");
  } else if (strcmp(command, "RIGHT") == 0) {
    setWarningState(STATE_RIGHT);
    Serial.println("ACK,RIGHT");
  } else if (strcmp(command, "CENTER") == 0) {
    setWarningState(STATE_CENTER);
    Serial.println("ACK,CENTER");
  } else if (strcmp(command, "BOTH") == 0) {
    setWarningState(STATE_BOTH);
    Serial.println("ACK,BOTH");
  } else if (strcmp(command, "SAFE") == 0 ||
             strcmp(command, "STOP") == 0) {
    setWarningState(STATE_SAFE);
    Serial.println("ACK,SAFE");
  } else if (strcmp(command, "PING") == 0) {
    Serial.println("PONG");
  } else if (strcmp(command, "STATUS") == 0) {
    sendStatus();
  } else if (command[0] != '\0') {
    Serial.print("ERR,UNKNOWN_COMMAND,");
    Serial.println(command);
  }
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    const char incoming = static_cast<char>(Serial.read());

    if (incoming == '\r') {
      continue;
    }

    if (incoming == '\n') {
      commandBuffer[commandLength] = '\0';
      if (commandLength > 0) {
        processCommand(commandBuffer);
      }
      commandLength = 0;
      continue;
    }

    if (commandLength < COMMAND_BUFFER_SIZE - 1) {
      commandBuffer[commandLength++] = incoming;
    } else {
      // Discard the overlong line to prevent partial command execution.
      commandLength = 0;
      Serial.println("ERR,COMMAND_TOO_LONG");
    }
  }
}

void updateWarningPattern() {
  if (currentState == STATE_SAFE) {
    return;
  }

  uint32_t onDuration = SIDE_ON_MS;
  uint32_t offDuration = SIDE_OFF_MS;

  if (currentState == STATE_CENTER) {
    onDuration = CENTER_ON_MS;
    offDuration = CENTER_OFF_MS;
  } else if (currentState == STATE_BOTH) {
    onDuration = BOTH_ON_MS;
    offDuration = BOTH_OFF_MS;
  }

  const uint32_t now = millis();
  const uint32_t interval = patternOn ? onDuration : offDuration;

  if (now - lastPatternChangeMs >= interval) {
    lastPatternChangeMs = now;
    patternOn = !patternOn;

    const bool leftOn = patternOn &&
      (currentState == STATE_LEFT || currentState == STATE_CENTER ||
       currentState == STATE_BOTH);
    const bool rightOn = patternOn &&
      (currentState == STATE_RIGHT || currentState == STATE_CENTER ||
       currentState == STATE_BOTH);

    writeOutputs(leftOn, rightOn, patternOn);
  }
}

void setup() {
  pinMode(LEFT_LED_PIN, OUTPUT);
  pinMode(RIGHT_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  writeOutputs(false, false, false);

  Serial.begin(SERIAL_BAUD);
  delay(250);

  Serial.println();
  Serial.println("VISENTRA_USB_BRIDGE,READY");
  Serial.println("BAUD,115200");
  Serial.println("COMMANDS,LEFT|RIGHT|CENTER|BOTH|SAFE|STOP|PING|STATUS");
  sendStatus();
}

void loop() {
  readSerialCommands();
  updateWarningPattern();
}

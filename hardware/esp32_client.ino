#include <Wire.h>
#include <WiFi.h>
#include <Arduino.h>
#include <AsyncTCP.h>
#include <SparkFun_ADXL345.h>
#include <ESPAsyncWebServer.h>

#define WARNING_LED 32
#define TAP_LED 25
#define TEST_LED 27

const char* ssid = "your_ssid";
const char* password = "your_password";

const float SAMPLE_RATE = 100.0; // 100 Hz
const unsigned long SAMPLE_INTERVAL = 1000000 / SAMPLE_RATE; // in microseconds
unsigned long lastSampleTime = 0;

AsyncWebServer server(80);
AsyncWebSocket ws("/ws");

ADXL345 adxl = ADXL345();

void ledBlink(int pin) {
    digitalWrite(pin, HIGH);
    vTaskDelay(100 / portTICK_PERIOD_MS);  
    digitalWrite(pin, LOW);
}

void onWebSocketEvent(AsyncWebSocket *server, AsyncWebSocketClient *client, AwsEventType type, void *arg, uint8_t *data, size_t len) {
    if (type == WS_EVT_CONNECT) {
        Serial.printf("Client connected: %u\n", client->id());
        ledBlink(TAP_LED);
    } 
    else if (type == WS_EVT_DISCONNECT) {
        Serial.printf("Client disconnected: %u\n", client->id());
        ledBlink(WARNING_LED);
    } 
    else if (type == WS_EVT_DATA) {
        String message = String((char*)data).substring(0, len);
        Serial.printf("Received: %s\n", message.c_str());
    }
}

void setup() {
    Serial.begin(115200);

    WiFi.setSleep(false);
    
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nConnected to WiFi");

    adxl.powerOn();
    adxl.setRate(100);
    adxl.setTapThreshold(30);
    adxl.setTapDuration(30);
    adxl.setDoubleTapLatency(60);
    adxl.setDoubleTapWindow(200);
    adxl.setTapDetectionOnXYZ(0, 0, 1);
    adxl.setInterrupt(ADXL345_INT_SINGLE_TAP_BIT, 1);
    adxl.setInterrupt(ADXL345_INT_DOUBLE_TAP_BIT, 1);
    adxl.setRangeSetting(1);

    pinMode(WARNING_LED, OUTPUT);
    pinMode(TAP_LED, OUTPUT);
    pinMode(TEST_LED, OUTPUT);

    ws.onEvent(onWebSocketEvent);
    server.addHandler(&ws);
    server.begin();
}

void sendPayload(int ax, int ay, int az, int tapStatus, int doubleTapStatus) {
    String payload = String(ax) + "," + String(ay) + "," + String(az) + "," + String(tapStatus) + "," + String(doubleTapStatus);
    ws.textAll(payload);  
    vTaskDelay(10 / portTICK_PERIOD_MS); 
}

void loop() {
    ws.cleanupClients();  
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("Reconnecting to WiFi...");
        WiFi.disconnect();
        WiFi.reconnect();
        delay(5000);
    }

    unsigned long currentTime = micros();
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
        lastSampleTime = currentTime;

        int x, y, z;
        adxl.readAccel(&x, &y, &z);

        byte interrupts = adxl.getInterruptSource();
        int tapStatus = 0;
        int doubleTapStatus = 0;

        if (adxl.triggered(interrupts, ADXL345_SINGLE_TAP)) {
            tapStatus = 1;
            digitalWrite(TAP_LED, HIGH);
            vTaskDelay(100 / portTICK_PERIOD_MS);  
            digitalWrite(TAP_LED, LOW);
        }

        if (adxl.triggered(interrupts, ADXL345_DOUBLE_TAP)) {
            doubleTapStatus = 1;
            digitalWrite(TEST_LED, HIGH);
            vTaskDelay(100 / portTICK_PERIOD_MS);  
            digitalWrite(TEST_LED, LOW);
        }

        sendPayload(x, y, z, tapStatus, doubleTapStatus);
    }
    vTaskDelay(10 / portTICK_PERIOD_MS);  
}

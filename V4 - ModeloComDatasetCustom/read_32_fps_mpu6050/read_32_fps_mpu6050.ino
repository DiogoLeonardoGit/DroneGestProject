#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <Adafruit_NeoPixel.h>

#define SDA_PIN 7  // GPIO7
#define SCL_PIN 6  // GPIO6
#define LED_PIN 8  // GPIO2 (LED on ESP32)

Adafruit_MPU6050 mpu;
const unsigned long interval = 1000 / 32;  // Interval between readings in milliseconds (1000ms / 32fps)
bool shouldRead = true;                    // Flag to control whether to read sensor data
unsigned long counter = 0;                 // Counter to keep track of prints

// Define NeoPixel settings
#define NUM_LEDS 1
Adafruit_NeoPixel pixels(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);

void setup(void) {
  Serial.begin(115200);
  while (!Serial)
    ;  // Wait for serial port to connect

  Wire.begin(SDA_PIN, SCL_PIN);  // Initialize I2C with the specified pins
  delay(2000);                   // Wait for the sensor to initialize

  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1)
      ;  // Wait indefinitely
  }
  Serial.println("MPU6050 Connected!");
}

void loop() {
  // Blink NeoPixel LED
  blinkNeoPixel();

  // Check for serial input
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    if (input.equals("stop")) {
      shouldRead = false;
      Serial.println("Reading stopped");
      Serial.println("counter: " + counter);
    } else if (input.equals("start")) {
      shouldRead = true;
      counter = 0;
      Serial.println("Reading started");
    }
  }

  if (shouldRead) {
    unsigned long startTime = millis();  // Record the start time of the loop

    // Read sensor data
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Send sensor values via serial port
    //Serial.print(counter);
    //Serial.print(",");
    Serial.print(millis());
    Serial.print(",");
    Serial.print(a.acceleration.x);
    Serial.print(",");
    Serial.print(a.acceleration.y);
    Serial.print(",");
    Serial.print(a.acceleration.z);
    Serial.print(",");
    Serial.print(g.gyro.x);
    Serial.print(",");
    Serial.print(g.gyro.y);
    Serial.print(",");
    Serial.println(g.gyro.z);

    // Increment the counter
    counter++;

    // Calculate time taken for the loop
    unsigned long elapsedTime = millis() - startTime;

    // Delay to maintain the desired interval between readings
    if (elapsedTime < interval) {
      delay(interval - elapsedTime);
    }
  }
}

// acender e apagar o led do arduino de 1 em 1 segundo
void blinkNeoPixel() {
  static unsigned long lastBlinkTime = 0;
  static bool ledState = false;

  unsigned long currentTime = millis();
  if (currentTime - lastBlinkTime >= 1000) {
    lastBlinkTime = currentTime;
    ledState = !ledState;

    // Set LED color to white with reduced intensity
    if (ledState) {
      pixels.setPixelColor(0, pixels.Color(10, 10, 10));  // White color with low intensity
    } else {
      pixels.setPixelColor(0, pixels.Color(0, 0, 0));  // Turn off LED
    }
    pixels.show();
  }
}

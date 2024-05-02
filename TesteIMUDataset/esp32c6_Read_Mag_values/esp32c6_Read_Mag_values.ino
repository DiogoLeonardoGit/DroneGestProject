#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <SPIFFS.h>

#define SDA_PIN 7  // GPIO7
#define SCL_PIN 6  // GPIO6

#define FILENAME "myDataset.csv" // File name to save data

Adafruit_MPU6050 mpu;

File dataFile;

void setup(void) {
  Serial.begin(115200);
  while (!Serial)
    delay(10); // will pause Zero, Leonardo, etc until serial console opens

  Wire.begin(SDA_PIN, SCL_PIN);  // Initialize I2C with specified pins
  delay(2000);  // Allow time for the sensor to initialize

  // Try to initialize MPU6050
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 Connected!");

  // initialize spiffs esp32 memory
  if (!SPIFFS.begin(true)) {
    Serial.println("SPIFFS initialization failed!");
    while (1) {
      delay(10);
    }
  }
  Serial.println("SPIFFS initialized.");

  // Check if the file exists
  if (!SPIFFS.exists(FILENAME)) {
      // If it doesn't exist, create it
      File dataFile = SPIFFS.open(FILENAME, FILE_WRITE);
      if (!dataFile) {
          Serial.println("Failed to create data file");
          return;
      }
      // Close the file after creating it
      dataFile.close();
  }

  // Open file for appending
  File dataFile = SPIFFS.open(FILENAME, FILE_APPEND);
  if (!dataFile) {
      Serial.println("Failed to open data file for appending");
      return;
  }

  // Write headers to the file if the file is empty
  if (dataFile.size() == 0) {
    dataFile.println("Time (ms), AccelX, AccelY, AccelZ, GyroX, GyroY, GyroZ");
  }

  // Set up motion detection
  mpu.setHighPassFilter(MPU6050_HIGHPASS_0_63_HZ);
  mpu.setMotionDetectionThreshold(1);
  mpu.setMotionDetectionDuration(10);
  mpu.setInterruptPinLatch(true);  // Keep it latched.  Will turn off when reinitialized.
  mpu.setInterruptPinPolarity(true);
  mpu.setMotionInterrupt(true);

  Serial.println("Ready to capture movement data...");
}

void loop() {
  if (mpu.getMotionInterruptStatus()) {
    /* Get new sensor events with the readings */
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    /* Print out the values */
    Serial.print("AccelX:");
    Serial.print(a.acceleration.x);
    Serial.print(",");
    Serial.print("AccelY:");
    Serial.print(a.acceleration.y);
    Serial.print(",");
    Serial.print("AccelZ:");
    Serial.print(a.acceleration.z);
    Serial.print(", ");
    Serial.print("GyroX:");
    Serial.print(g.gyro.x);
    Serial.print(",");
    Serial.print("GyroY:");
    Serial.print(g.gyro.y);
    Serial.print(",");
    Serial.print("GyroZ:");
    Serial.print(g.gyro.z);
    Serial.println("");

    // Write data to SPIFFS card
    dataFile.print(millis());
    dataFile.print(",");
    dataFile.print(a.acceleration.x);
    dataFile.print(",");
    dataFile.print(a.acceleration.y);
    dataFile.print(",");
    dataFile.print(a.acceleration.z);
    dataFile.print(",");
    dataFile.print(g.gyro.x);
    dataFile.print(",");
    dataFile.print(g.gyro.y);
    dataFile.print(",");
    dataFile.println(g.gyro.z);
    dataFile.flush(); // Ensure data is written to the file immediately
  }

  // Delay for a short period to allow for other tasks
  delay(10);
}

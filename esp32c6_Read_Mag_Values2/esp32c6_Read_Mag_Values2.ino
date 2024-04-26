#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

#define SDA_PIN 7  // GPIO7
#define SCL_PIN 6  // GPIO6

Adafruit_MPU6050 mpu;

void setup(void) {
  Serial.begin(115200);
  while (!Serial)
    delay(10); // Pausa até que a conexão serial seja estabelecida

  Wire.begin(SDA_PIN, SCL_PIN);  // Inicializa I2C com os pinos especificados
  delay(2000);  // Aguarda a inicialização do sensor

  if (!mpu.begin()) {
    Serial.println("Falha ao encontrar o chip MPU6050");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 Conectado!");
}

void loop() {
  if (mpu.getMotionInterruptStatus()) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Envia os valores dos sensores via porta serial
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
  }

  // Atraso para permitir outras tarefas
  delay(10);
}

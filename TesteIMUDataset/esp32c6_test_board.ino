#include <Adafruit_NeoPixel.h>

#define noOfPixels 1
#define rgbPin 8

Adafruit_NeoPixel rgbLed(noOfPixels, rgbPin, NEO_GRB + NEO_KHZ800);

bool canRun = true;

void setup() {
  Serial.begin(115200);
  rgbLed.begin();
  rgbLed.show(); // Initialize all pixels to 'off'
  Serial.println("Test");
}

void loop() {
  // Se não podemos rodar, saímos do loop
  if (!canRun) {
    return;
  }

  Serial.println("Test");

  // Define a cor branca e exibe-a
  rgbLed.setPixelColor(0, rgbLed.Color(255, 255, 255));
  rgbLed.show();
  delay(2000);

  // Define a cor verde e exibe-a
  rgbLed.setPixelColor(0, rgbLed.Color(0, 100, 0));
  rgbLed.show();
  delay(2000);

  // Define a cor magenta e exibe-a
  rgbLed.setPixelColor(0, rgbLed.Color(255, 0, 255));
  rgbLed.show();
  delay(2000);
}

void test() {
  // Inverte o estado da variável canRun
  canRun = !canRun;
}

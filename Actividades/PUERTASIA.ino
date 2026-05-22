#include <Servo.h>

Servo servo1; // Pin 9
Servo servo2; // Pin 10

void setup() {
  Serial.begin(9600);
  servo1.attach(9);
  servo2.attach(10);
  
  servo1.write(0);
  servo2.write(0);
}

void loop() {
  if (Serial.available() > 0) {
    
    char comando = Serial.read();
    
    if (comando == '\n' || comando == '\r') {
      return; 
    }
    
    if (comando == '1') {        
      servo2.write(0); //  cierre inmediato del otro para que no actúen juntos
      servo1.write(90);          
    } 
    else if (comando == '3') {   
      servo1.write(0); 
      servo2.write(90);          
    } 
    else if (comando == '0') {   
      servo1.write(0);           
      servo2.write(0);
    }
    
    while(Serial.available() > 0) {
      Serial.read(); 
    }
  }
}
#include <Servo.h>
Servo servoMain1;
Servo servoMain2;
int EN1 = 5;
int EN2 = 6;
int IN1 = 4;
int IN2 = 7;
int alarmPin = 8;
int pos = 0;

void setup() {
  pinMode(EN1, OUTPUT);
  pinMode(EN2, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(alarmPin, OUTPUT);
  servoMain1.attach(9);
  servoMain2.attach(11);
  Serial.begin(9600);
}

void loop() {
    servoh();
  if (Serial.available() > 0) {
    char data = Serial.read();
  if (data == 'F') {
    forward();
    alarm();
  } else if (data == 'S') {
    stopMotors();
    stopAlarm();
  } else if (data == 'L') {
    left();
    alarm();
  } else if (data == 'R') {
    right();
    alarm();
    }
  }
}
void forward() {
  analogWrite(EN1, 200);
  analogWrite(EN2, 200);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
}
void right() {
  analogWrite(EN1, 100);
  analogWrite(EN2, 100);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
}
void left() {
  analogWrite(EN1, 100);
  analogWrite(EN2, 100);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
}
void stopMotors() {
  digitalWrite(EN1, LOW);
  digitalWrite(EN2, LOW);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
}
void alarm() {
  digitalWrite(alarmPin, HIGH);
}
void stopAlarm() {
  digitalWrite(alarmPin, LOW);
}
void servoh() {
  servoMain1.write(90);
  servoMain2.write(90);
  servoMain1.write(180);
  servoMain2.write(180);
  servoMain1.write(90);
  servoMain2.write(90);
  servoMain1.write(0);
  servoMain2.write(0);
}

#include "Arduino.h"
#include "MecanumDriver.h"

MecanumDriver::MecanumDriver(
        int pin1, int pin2, int pin3, int pin4,
        int pin5, int pin6, int pin7, int pin8
):motor1_pin1(pin1), motor1_pin2(pin2), motor2_pin1(pin3), motor2_pin2(pin4),
    motor3_pin1(pin5), motor3_pin2(pin6), motor4_pin1(pin7), motor4_pin2(pin8) {
}

MecanumDriver::~MecanumDriver() {}

void MecanumDriver::begin() {
    pinMode(motor1_pin1, OUTPUT);
    pinMode(motor1_pin2, OUTPUT);
    pinMode(motor2_pin1, OUTPUT);
    pinMode(motor2_pin2, OUTPUT);
    pinMode(motor3_pin1, OUTPUT);
    pinMode(motor3_pin2, OUTPUT);
    pinMode(motor4_pin1, OUTPUT);
    pinMode(motor4_pin2, OUTPUT);

    driveAllMotor(0, 0, 0, 0);
}

void MecanumDriver::driveMotor(int motor, int speed) {
    int pin1, pin2, direction;
    switch (motor) {
        case 1:
            pin1 = motor1_pin1;
            pin2 = motor1_pin2;
            direction = motor1_direction;
            break;
        case 2:
            pin1 = motor2_pin1;
            pin2 = motor2_pin2;
            direction = motor2_direction;
            break;
        case 3:
            pin1 = motor3_pin1;
            pin2 = motor3_pin2;
            direction = motor3_direction;
            break;
        case 4:
            pin1 = motor4_pin1;
            pin2 = motor4_pin2;
            direction = motor4_direction;
            break;
        default:
            return;
    }
    if (direction >= 0) {
        driveMotor(pin1, pin2, speed);
    } else {
        driveMotor(pin2, pin1, speed);
    }
}

void MecanumDriver::driveMotor(int pin1, int pin2, int speed) {
    if (speed >= 0) {
        analogWrite(pin1, constrain(abs(speed), 0, 100));
        analogWrite(pin2, 0);
    } else {
        analogWrite(pin1, 0);
        analogWrite(pin2, constrain(abs(speed), 0, 100));
    }
}

void MecanumDriver::driveAllMotor(int speed1, int speed2, int speed3, int speed4) {
    driveMotor(1, speed1);
    driveMotor(2, speed2);
    driveMotor(3, speed3);
    driveMotor(4, speed4);
}

void MecanumDriver::setDirection(int motor, int direction) {
    switch (motor) {
        case 1:
            motor1_direction = direction;
            break;
        case 2:
            motor2_direction = direction;
            break;
        case 3:
            motor3_direction = direction;
            break;
        case 4:
            motor4_direction = direction;
            break;
        default:
            return;
    }
}

void MecanumDriver::setAllDirection(int direction1, int direction2, int direction3, int direction4) {
    setDirection(1, direction1);
    setDirection(2, direction2);
    setDirection(3, direction3);
    setDirection(4, direction4);
}
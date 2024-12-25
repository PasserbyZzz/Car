#ifndef MECANUM_DRIVER_H
#define MECANUM_DRIVER_H

class MecanumDriver {
public:
    MecanumDriver(int pin1, int pin2, int pin3, int pin4, int pin5, int pin6, int pin7, int pin8);

    ~MecanumDriver();

    void begin(void);

    void driveMotor(int motor, int speed);

    void driveAllMotor(int speed1, int speed2, int speed3, int speed4);

    void setDirection(int motor, int direction);

    void setAllDirection(int direction1, int direction2, int direction3, int direction4);

private:
    int motor1_pin1;
    int motor1_pin2;
    int motor1_direction = 1;

    int motor2_pin1;
    int motor2_pin2;
    int motor2_direction = -1;

    int motor3_pin1;
    int motor3_pin2;
    int motor3_direction = 1;

    int motor4_pin1;
    int motor4_pin2;
    int motor4_direction = -1;

    void driveMotor(int pin1, int pin2, int speed);
};

#endif
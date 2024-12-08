#ifndef MECANUM_DRIVER_H
#define MECANUM_DRIVER_H

class MecanumDriver {
public:
    // 创建电机驱动器对象，pin1~pin8对应驱动板IN1~IN8
    MecanumDriver(int pin1, int pin2, int pin3, int pin4, int pin5, int pin6, int pin7, int pin8);

    ~MecanumDriver();

    // 开启驱动器，自动完成引脚配置
    void begin(void);

    // 驱动单个电机，参数motor可取1~4，speed范围为-255~255，符号代表转向
    void driveMotor(int motor, int speed);

    // 同时驱动4个电机
    void driveAllMotor(int speed1, int speed2, int speed3, int speed4);

    // 设置单个电机转向的正方向，参数motor可取1~4，direction正负决定正方向
    void setDirection(int motor, int direction);

    // 设置4个电机转向的正方向
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
电机编号：
1    ↑    2
      ↑
      ↑
3    ↑    4

连线：
OUT1/OUT3/OUT5/OUT7：电机1/2/3/4红线
OUT2/OUT4/OUT6/OUT8：电机1/2/3/4黑线

函数：
1.MecanumDriver(int pin1, int pin2, int pin3, int pin4, int pin5, int pin6, int pin7, int pin8);
创建电机驱动器对象，pin1~pin8对应驱动板IN1~IN8
2.void begin(void);
开启驱动器，自动完成引脚配置
3.void driveMotor(int motor, int speed);
驱动单个电机，参数motor可取1~4，speed范围为-255~255，符号代表转向
4.void driveAllMotor(int speed1, int speed2, int speed3, int speed4);
同时驱动4个电机
5.void setDirection(int motor, int direction);
设置单个电机转向的正方向，参数motor可取1~4，direction正负决定正方向
6.void setAllDirection(int direction1, int direction2, int direction3, int direction4);
设置4个电机转向的正方向
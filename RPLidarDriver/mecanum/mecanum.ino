#include "MecanumDriver.h"
MecanumDriver mecanum = MecanumDriver(9, 8, 12, 13, 11, 10, 46, 21);

void setup() {
  mecanum.begin();
}

void loop() {
  // 前
  mecanum.driveAllMotor(100, 100, 100, 100);//屎
  delay(1000);
  // 后
  mecanum.driveAllMotor(-100, -100, -100, -100);
  delay(1000);
  // 左
  mecanum.driveAllMotor(-100, 100, 100, -100);
  delay(1000);
  // 右
  mecanum.driveAllMotor(100, -100, -100, 100);
  delay(1000);
  // 
  mecanum.driveAllMotor(100, 0, 0, 0);
  delay(1000);
  mecanum.driveAllMotor(-100, 0, 0, 0);
  delay(1000);  
  mecanum.driveAllMotor(0, 100, 0, 0);
  delay(1000);
  mecanum.driveAllMotor(0, -100, 0, 0);
  delay(1000);
  mecanum.driveAllMotor(0, 0, 100, 0);
  delay(1000);
  mecanum.driveAllMotor(0, 0, -100, 0);
  delay(1000);
  mecanum.driveAllMotor(0, 0, 0, 100);
  delay(1000);
  mecanum.driveAllMotor(0, 0, 0, -100);
  delay(1000);

  // 转45度要多长时间
  // mecanum.driveAllMotor(100, -100, 100, -100);
  // delay(40);
  // mecanum.driveAllMotor(0, 0, 0, 0);
  // delay(1000);
  // // 转180度要多长时间
  // mecanum.driveAllMotor(100, -100, 100, -100);
  // delay(650);
  // mecanum.driveAllMotor(0, 0, 0, 0);
  // delay(1000);
}

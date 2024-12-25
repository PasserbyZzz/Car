#include <RPLidar.h>
#include "MecanumDriver.h" 

// 创建激光雷达对象
RPLidar lidar;
// 创建麦克纳姆轮驱动器对象
MecanumDriver mecanum(9, 8, 12, 13, 11, 10, 46, 21);
// 用于存储激光雷达数据的数组（0到359度共360个值）
float distances[360] = { 0 };


void setup() {
  Serial.begin(115200);  // 开启用于调试的串口
  lidar.begin(Serial2);  // 将激光雷达连接到串口2（19、20引脚）
  mecanum.begin();  // 启动麦克纳姆轮驱动器
}

void turnLeft(int speed) {
    mecanum.driveAllMotor(-speed, speed, -speed, speed);
}

void turnRight(int speed) {
    mecanum.driveAllMotor(speed, -speed/3, speed, -speed/3);
}

void goStraight(int speed, float right) {
    if (right > 0.6) {
      // mecanum.driveAllMotor(speed/2, -speed/2, -speed/2, speed/2); // right
      mecanum.driveAllMotor(speed - 10, speed, speed - 10, speed);
    }
    if (right < 0.5) {
      mecanum.driveAllMotor(speed, speed - 10, speed, speed - 10);
    } 
    else {
      mecanum.driveAllMotor(speed, speed, speed, speed);
    }

    // mecanum.driveAllMotor(speed, speed, speed, speed);
    // float diviation = left - right;
    // float offset = 0.1;
    // if (diviation < -offset) {
    //   // turn left
    //   mecanum.driveAllMotor(speed, speed - 20, speed, speed - 20);
    // }
    // else if (diviation > offset) {
    //   // turn right
    //   mecanum.driveAllMotor(speed - 20, speed, speed - 20, speed);
    // }
    // else {
    //   mecanum.driveAllMotor(speed, speed, speed, speed); // forward
    // }
}

void loop() {
  if (IS_OK(lidar.waitPoint())) {                                // 等到一个新的扫描点
    float distance = lidar.getCurrentPoint().distance / 1000.0;  // 距离值，单位m
    int angle = lidar.getCurrentPoint().angle;                   // 角度值（整数，四舍五入）
    bool startBit = lidar.getCurrentPoint().startBit;            // 每进入一次新的扫描时为true，其它时候为false
    if (angle >= 0 && angle < 360) {                             // 角度值在[0, 359]范围内
      distances[angle] = distance;                               // 将距离值存储到数组
    }

    if (startBit) {             // 每进入一次新的扫描处理并控制一次
      float distance_min = 10;  // 用于存储最近障碍物距离
      int angle_min = 0;        // 用于存储最近障碍物对应角度
      // 遍历寻找[0, 359]范围内最近障碍物距离及其对应角度
      for (int angle = 0; angle < 360; angle++) {
        float distance = distances[angle];
        if (distance >= 0.15) {           // 激光雷达的最小量程为0.15m，>=0.15的才是有效数据
          if (distance < distance_min) {  // 如果障碍物距离<最近距离
            distance_min = distance;      // 更新障碍物最近距离
            angle_min = angle;            // 更新最近障碍物对应角度
          }
        }
      }

      // 收集四个方向的距离数据
      int start = -22, stop = 22;
      int num = stop - start + 1;
      int num_forward = 0, num_back = 0, num_left = 0, num_right = 0;
      float distance_threshold = 0.6;
      float forward = 0, back = 0, left = 0, right = 0;
      for (int angle = start; angle <= stop; angle++) {
        right += distances[270 + angle];
        if (distances[180 + angle] >= distance_threshold) {
          num_forward++;
        }
        if (distances[angle%360] >= distance_threshold) {
          num_back++;
        }
        if (distances[90 + angle] >= distance_threshold) {
          num_left++;
        }
        if (distances[270 + angle] >= distance_threshold) {
          num_right++;
        }
        // forward += distances[180 + angle];
        // back += distances[angle % 360];
        // left += distances[90 + angle];
        // right += distances[270 + angle];
      }
      right /= num;

      int num_threshold1 = 10;
      int num_threshold2 = 7;
      bool is_forward, is_back, is_left ,is_right;
      is_forward = num_forward >= num_threshold2 ? true : false;
      // is_back = num_back >= num_threshold1 ? true : false;
      is_left = num_left >= num_threshold2 ? true : false;
      is_right = num_right >= num_threshold1 ? true : false;

      // the value of threshold needs to be adjusted in the process of practice.
      // float threshold = 0.5;
      int speed = 80;

      if (is_right) {
        Serial.println("Turning right");
        Serial.println(num_right);
        turnRight(speed);
      }
      else if (is_forward) {
        Serial.println("Going straight");
        Serial.println(num_forward);
        goStraight(speed, right);
      }
      else if (is_left) {
        Serial.println("Turning left");
        Serial.println(num_forward);
        turnLeft(speed);
      }
      else {
        Serial.println("Turning right2");
        turnRight(speed);

      }

      // if (forward > threshold) {
      //   Serial.println("Going straight");
      //   goStraight(speed, left, right);
      // }

      // else if (right > threshold) {
      //   Serial.println("Turning right");
      //   turnRight(speed);
      // }

      // else if (left > threshold) {
      //   Serial.println("Turning left");
      //   turnLeft(speed);
      // }

    }
  } else {
    // 重新连接激光雷达
    rplidar_response_device_info_t info;
    if (IS_OK(lidar.getDeviceInfo(info, 100))) {
      lidar.startScan();
      delay(1000);
    }
  }
}

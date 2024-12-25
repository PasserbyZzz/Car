#include <RPLidar.h>
#include <cmath>
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

void setspeed(float x, float y, float w)//正： x向前，y向左，w逆时针。
{
  float fl,fr,bl,br;
  fl = x + y - w;
  fr = x - y + w;
  bl = x - y - w;
  br = x + y + w;

  mecanum.driveAllMotor(fl, fr, bl, br);
}

float midline(float data[])
{
  int num = 0;
  float res = 0;
  for(int i = 22.5 + 45 * 1; i < 22.5 + 45 * 2; i++)
  {
    if(data[i] != 0)
    {
      res += (data[i] - data[360 - i]);
      num ++;
    }
  }
  return res/num;
}

float getFront(float data[])
{
  int num = 0;
  float res = 0;
  for(int i = 22.5 + 45 * 3; i < 22.5 + 45 * 4; i++)
  {
    if(data[i] != 0)
    {
      res += data[i];
      num ++;
    }
  }
  return res/num;
}

float getRight(float data[])
{
  int num = 0;
  float res = 0;
  for(int i = 22.5 + 45 * 5 + 22.5; i < 22.5 + 45 * 6; i++)
  {
    if(data[i] != 0)
    {
      res += data[i];
      num ++;
    }
  }
  return res/num;
}

float getLeft(float data[])
{
  int num = 0;
  float res = 0;
  for(int i = 22.5 + 45 * 1; i < 45 * 2; i++)
  {
    if(data[i] != 0)
    {
      res += data[i];
      num ++;
    }
  }
  return res/num;
}

float getAround(float data[])
{
  int num = 0;
  float res = 0;
  for(int i = 22.5 + 45 * 1; i < 22.5 + 45 * 6; i++)
  {
    if(data[i] != 0)
    {
      res += data[i];
      num ++;
    }
  }
  return res/num;
}

float mid = 0.0;
float last_mid = 0.0;
float k = 80; 
float p = -10; 

bool flag;

float x_speed = 60;
float w = 0.0;

void loop() {
  if (IS_OK(lidar.waitPoint())) {                                // 等到一个新的扫描点
    float distance = lidar.getCurrentPoint().distance;           // 距离值，单位mm
    int angle = lidar.getCurrentPoint().angle;                   // 角度值（整数，四舍五入）
    bool startBit = lidar.getCurrentPoint().startBit;            // 每进入一次新的扫描时为true，其它时候为false
    if (angle >= 0 && angle < 360) {                             // 角度值在[0, 359]范围内
      distances[angle] = distance;                               // 将距离值存储到数组
    }

    if (startBit) {             // 每进入一次新的扫描处理并控制一次
      // 优先右转
      // 不能右转则直行
      // 不能直行则左转
      // 不能左转则掉头

      last_mid = mid;
      mid = midline(distances);

      float dmid = last_mid - mid;

      w = (k * mid + p * dmid) / 1000.00;
      // w = k * mid / 1000.00;

      flag = true;
      // 优先右转
      if (getRight(distances) > 750)
      {
        // Serial.print(getRight(distances));
        Serial.println("RIGHT");

        // if (w < -100)
        //   setspeed(0, 0, -200);
        // else
        //   setspeed(0, 0, w);

        mecanum.driveAllMotor(100, -100, 100, -100);
        // delay(40);

        flag = false;
      }

      // 不能右转则直行(默认巡线)
      else if (getFront(distances) > 500)
      { 
        // Serial.print(getFront(distances));
        // Serial.println("STRAIGHT");
      } 

      // 不能直行则左转
      else if (getLeft(distances) > 750)
      {
        // Serial.print(getLeft(distances));
        Serial.println("LEFT");

        // if (w > 100)
        //   setspeed(0, 0, 200);
        // else
        //   setspeed(0, 0, w);

        mecanum.driveAllMotor(-100, 100, -100, 100);
        // delay(40);

        flag = false;
      }

      // 不能左转则掉头
      else if (getAround(distances) < 450)
      {
        // Serial.println("AROUND");

        mecanum.driveAllMotor(100, -100, 100, -100);
        delay(650);

        flag = false;
      }

      // if (mid > 0)
      //   x_speed -= (mid / 100.00);
      // if (mid < 0)
      //   x_speed += (mid / 100.00);

      // if(mid <= 200 && mid >= 0)
      //   {
      //     x_speed += 0.5 * (200 - mid);
      //   }

      // if(mid >= -200 && mid <= 0)
      // {
      //   x_speed += 0.5 * (200 + mid);
      // }

      Serial.print("Left: ");
      Serial.print(getLeft(distances));
      Serial.print(" Right: ");
      Serial.print(getRight(distances));
      Serial.print(" Front: ");
      Serial.print(getFront(distances));
      Serial.print(" Around: ");
      Serial.print(getAround(distances));
      Serial.print(" Mid: ");
      Serial.println(mid);
      // Serial.print(" x_speed: ");
      // Serial.print(x_speed);
      // Serial.print(" w: ");
      // Serial.println(w);

      // if (flag)
      // {
      //   if (abs(w) > 5 && abs(w) < 100)
      //     setspeed(x_speed, 0, w);
      //   else
      //     setspeed(x_speed, 0, 0);
      // }

      if (flag)
      {
        Serial.println("STRAIGHT");
        if (mid > 60) 
        {
          Serial.print("LEFT");
          mecanum.driveAllMotor(x_speed - 20, x_speed + 10, x_speed - 20, x_speed + 10);
        }
        if (mid < 0) 
        {
          Serial.print("RIGHT");
          mecanum.driveAllMotor(x_speed + 10, x_speed - 20, x_speed + 10, x_speed - 20);
        } 
        else  
        {
          Serial.println("NO NEED");
          mecanum.driveAllMotor(x_speed, x_speed, x_speed, x_speed);
        }
      }
      


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

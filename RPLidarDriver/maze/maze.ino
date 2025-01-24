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
  float res = 0;
  for(int i = 225;i < 255; i++)
  {
    res += (data[i] - data[360 - i]);
  }

  return res/30.00 - 50.0;

}

float getfront(float data[])
{
  float res = 0;
  int num = 0;
  for(int i = 170; i < 190; i++)
  {
    if(data[i] != 0)
    {
      res += data[i];
      num ++;
    }
  }

  return res/(num + 0.01);
}

float getrightfront(float data[])
{
  float res = 0;
  int num = 0;
  for(int i = 225; i < 265; i++)
  {
    if(data[i] != 0)
    {
      res += data[i];
      num ++;
    }
  }

  return res/(num + 0.01);
}

float getrightback(float data[])
{
  float res = 0;
 int num = 0;
  for(int i =275; i < 315; i++)
  {
    if(data[i] != 0)
    {
      res += data[i];
      num ++;
    }
  }

  return res/(num + 0.01);//防止nan的错误

}

float difference = 0.0;
float last_difference = 0.0;
float k = 80.20; //80 90 40
float p = -16.00; //50 50 40


float w = 0.0;
      
float threshold = 200.00;

void loop() {
  float x_speed = 50.00;
  float y_speed = 0.00;

  if (IS_OK(lidar.waitPoint())) {                                // 等到一个新的扫描点
    float distance = lidar.getCurrentPoint().distance;           // 距离值，单位m
    int angle = round(lidar.getCurrentPoint().angle);            // 角度值（整数，四舍五入）
    bool startBit = lidar.getCurrentPoint().startBit;            // 每进入一次新的扫描时为true，其它时候为false
    if (angle >= 0 && angle < 360) {                             // 角度值在[0, 359]范围内                          // 将距离值存储到数
        distances[angle] = distance;
    }
  
    if(startBit)
    {
      last_difference = difference;
      float rf = getrightfront(distances);
      float rb = getrightback(distances);
      float d_difference = difference - last_difference;
      difference = rf - rb;
      //difference = midline(distances);
      float mid = (rb + rf)/2;

      w = -(k*difference + p*d_difference) / 200.00;
      
      // Serial.print("front:");
      // Serial.println(rf);     
      // Serial.print("back:");
      // Serial.println(rb);
      // Serial.print("d:");
      // Serial.println(difference);
      // Serial.print("front:");
      // Serial.println(getfront(distances) + 0.01);
      // Serial.println(getfront(distances) < threshold);
      
      if (getfront(distances) < threshold)
      {
        x_speed = -35;  
        w = w > 0 ? 30.00 * w : w / 8.00;
      }
      
      else if (mid < 250.00)
      {
        // Serial.println("getfront(distances) < threshold");
        y_speed = -55.00;
      }



      else if (difference > 800.00)
      {
        w /= 8.00;
      }
      
      if(getfront(distances) >= 800.00)
      {
        x_speed += 60;
      }

      setspeed(x_speed, y_speed, w);
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

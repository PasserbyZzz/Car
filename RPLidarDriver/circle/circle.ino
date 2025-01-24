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

float mid = 0.0;
float last_mid = 0.0;
float k = 40.20; //80 90 40
float p = -35.00; //50 50 40

float x_speed = 27.50;
float w = 0.0;

void loop() {

  if (IS_OK(lidar.waitPoint())) {                                // 等到一个新的扫描点
    float distance = lidar.getCurrentPoint().distance;  // 距离值，单位m
    int angle = lidar.getCurrentPoint().angle;                   // 角度值（整数，四舍五入）
    bool startBit = lidar.getCurrentPoint().startBit;            // 每进入一次新的扫描时为true，其它时候为false
    if (angle >= 0 && angle < 360) {                             // 角度值在[0, 359]范围内
      distances[angle] = distance;                               // 将距离值存储到数组
    }
  
    if(startBit)
    {
      last_mid = mid;
      mid = midline(distances);

      float dmid = last_mid - mid;

      w = (-k*mid - p*dmid) / 1000.00;

      Serial.print("mid:");
      Serial.print(mid);
      
      Serial.print("   w:");
      Serial.println(w);

      // if(mid <= 200 && mid >= 0)
      //   {
      //     x_speed += 0.5 * (200 - mid);
      //   }

      // if(mid >= -200 && mid <= 0)
      // {
      //   x_speed += 0.5 * (200 + mid);
      // }
    }
    setspeed(x_speed, 0, w);

  } else {
    // 重新连接激光雷达
    rplidar_response_device_info_t info;
    if (IS_OK(lidar.getDeviceInfo(info, 100))) {
    lidar.startScan();
    delay(1000);
    }  
  }
}

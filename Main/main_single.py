# coding:utf-8
# 加入摄像头模块，让小车实现自动循迹行驶
# 思路为:摄像头读取图像，进行二值化，将白色的赛道凸显出来
# 选择下方的一行像素，红色为 0
# 找到0值的中点
# 目标中点与标准中点(320)进行比较得出偏移量
# 根据偏移量来控制小车左右轮的转速
# 考虑了偏移过多失控->停止;偏移量在一定范围内->高速直行(这样会速度不稳定，已删)

# import pandas as pd
# from scipy import linalg
# import tflite_runtime.interpreter as tflite
# import threading  
# import threading  # 导入 threading 库

import cv2
import numpy as np
from new_driver import driver
import time
from threading import Thread
import torch
import torchvision.transforms as transforms
from PIL import Image


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=20):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        #ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        #ret = self.stream.set(3,resolution[0])
        #ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

car = driver()
# 定义看到crosswalk的次数
crosswalk_num = 0
# 初始化输入网络的图像尺寸
image_size=(120,160)
# 打开摄像头，图像尺寸 640*480(长*高)，opencv 存储值为 480*640(行*列) 
videostream = VideoStream(resolution=(480,640),framerate=10).start()
time.sleep(1)


def is_white(point):
    if point[1] < 20  and point[2] > 230:
        return 1
    else:
        return 0
    
def is_yellow(point):
    if point[0] > 26 and point[0] < 34 and point[1] >43 and point[2] > 46:
        return 1
    else:
        return 0
    
def getmid(hsv):
 
    midline = []
    for y in range(80, 100):
        white_x = [81]
        for x in range(20,140):
            if is_white(hsv[y][x]):
                white_x.append(x)
                hsv[y][x] = (0, 0, 0)
        if(len(white_x) == 0):
            pass
        else:
            midline.append(sum(white_x)/len(white_x))
            #midline.append((white_x[-1] + yellow_x[-1])/2)
        hsv[y][int(midline[-1])] = (0, 0, 0)
    
    
    return sum(midline)/len(midline)                  
        
    

# upload calibration matrix
# data = np.load('calibration.npz')
# cameraMatrix = data['cameraMatrix']
# distCoeffs = data['distCoeffs']

try:
    
    last_dmid = 0
    dmid = 0
    while True:

        frame = videostream.read()

#         frame = cv2.undistort(frame, cameraMatrix, distCoeffs, None, cameraMatrix)

        frame = cv2.resize(frame, None, fx = 0.25, fy = 0.25, interpolation= cv2.INTER_NEAREST) #采样 160*120
        
        HSV_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #print(type(HSV_frame), HSV_frame[30][30])

        
        
        
        # 按q键可以退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                               
        
        mid = getmid(HSV_frame)
        
        kp = 2
        kd = 1
        
        
        #cv2.imshow("frame",HSV_frame)
        
        last_mid = dmid
        dmid = 81 - mid
        
        d = dmid - last_mid
          
        w = kp * dmid + kd * d
        
        x_speed = 80
        y_speed = 0
        
        

        
        if(abs(dmid) >= 18):
            x_speed /= 5
            
            
        if(abs(dmid) <= 5):
           x_speed *= -0.125*abs(2*dmid) + 2.25
            
        #print(mid,"|",dmid)
        car.set_speed(x_speed, y_speed, w)
        
        #print(w)
        
        

finally:
    # 确保在程序退出前停止小车
    print("Stopping the vehicle...")
    car.set_speed(0, 0, 0)
    videostream.stop()
    cv2.destroyAllWindows()


        


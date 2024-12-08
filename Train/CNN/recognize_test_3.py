import cv2
import torch
import time
from threading import Thread
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
from new_driver import driver

# 定义与训练时相同的 CNN 模型
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.Conv2d(16, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32 * 4 * 4)
        x = self.fc(x)
        return x

# 加载模型
num_classes = 4  # 类别数量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(num_classes).to(device)
model.load_state_dict(torch.load('traffic_sign_model.pth'))
model.double()
model.eval()


# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 类别标签（与训练时保持一致）
class_labels = ['left', 'park', 'right', 'straight']

# 打开摄像头

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
videostream = VideoStream(resolution=(480,640),framerate=10).start()
time.sleep(1)

def is_white(point):
    if point[1] < 20  and point[2] > 230:
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
        hsv[y][int(midline[-1])] = (0, 0, 0)

    return sum(midline)/len(midline)                  
        
car = driver()
last_dmid = 0
dmid = 0

while True:

    frame = videostream.read()
    
    frame_c = frame.deepcopy()

    frame_c = cv2.resize(frame_c, None, fx = 0.25, fy = 0.25, interpolation= cv2.INTER_NEAREST) #采样 160*120

    # 转换为 HSV 格式以检测蓝色区域
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_frame = cv2.cvtColor(frame_c, cv2.COLOR_BGR2HSV)
    mid = getmid(hsv_frame) 

    kp = 2
    kd = 1

    last_mid = dmid
    dmid = 81 - mid

    d = dmid - last_mid

    w = kp * dmid + kd * d

    x_speed = 40
    y_speed = 0

    if(abs(dmid) >= 18):
        x_speed /= 5
    if(abs(dmid) <= 5):
       x_speed *= -0.125*abs(2*dmid) + 2.25

    print(mid,"|",dmid)
    '''
    这里为什么注释了？
    '''
    #car.set_speed(x_speed, y_speed, w) 
    print(w)
    
    # 蓝色的 HSV 值范围（可以根据实际情况调整）
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    
    # 创建蓝色掩码
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # 寻找轮廓来确定蓝色区域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    '''
    这里的逻辑需要改一下
    '''
    max_cnt = 0
    max_size = 0
    x, y, w1, h = 0, 0, 0, 0
    
    # 找区域面积最大的轮廓
    for cnt in contours:
        x, y, w1, h = cv2.boundingRect(cnt)
        if w1 * h > max_size:
            max_cnt = cnt
            '''
            这里w忘记改了
            '''
            max_size = w1 * h 
    
    if max_cnt != 0: # 能够找到蓝色区域
        x, y, w1, h = cv2.boundingRect(max_cnt)
    print(w1, ",", h)
    # 设置最小尺寸过滤噪声
    predicted_class = 'straight'
    '''
    这里60会不会太大了
    '''
    if w1 > 60 and h > 60: 
        # 画出边界框
        cv2.rectangle(frame, (x, y), (x + w1, y + h), (0, 255, 0), 2)
        
        # 提取蓝色区域并预处理为模型输入格式
        sign_img = frame[y:y+h, x:x+w1]
        sign_img = cv2.cvtColor(sign_img, cv2.COLOR_BGR2RGB)
        sign_img = Image.fromarray(sign_img)
        sign_img = transform(sign_img).unsqueeze(0).to(device)  # 增加批次维度

        # 将图像输入模型并预测类别
        with torch.no_grad():
            outputs = model(sign_img.double())
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
            predicted_class = class_labels[predicted.item()]
            confidence = probabilities[0][predicted.item()].item()

        # 显示预测结果
        cv2.putText(frame, f"{predicted_class}: {confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 显示处理后的帧
    cv2.imshow('Traffic Sign Recognition', frame)
    
    # 默认巡线
    if predicted_class == 'straight':
        car.set_speed(x_speed, y_speed, w)
    
    # 左转
    elif predicted_class == 'left':
        car.set_speed(0, 0, 40)
        '''
        这里可以加一个转向时间（1s），使小车充分转向
        '''
        time.sleep(1)
        
    # 右转
    elif predicted_class == 'right':
        car.set_speed(0, 0, -40)
        '''
        这里可以加一个转向时间（1s），使小车充分转向
        '''
        time.sleep(1)
        
    # 停车
    elif predicted_class == 'park':
        car.set_speed(0, 0, 0)
        '''
        这里要不要退出循环，使小车一直停？
        '''
        
    # 默认巡线
    else:
        car.set_speed(x_speed, y_speed, w)

    time.sleep(0.05)
    
    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Stopping the vehicle...")
car.set_speed(0, 0, 0)
videostream.stop()
cv2.destroyAllWindows()

'''
1. 下节课可以测试一下画面的别的区域会不会出现蓝色区域的干扰，
按道理应该不会有影响；如果会，则可以在框选蓝色区域的时候选择画面的正中间

2. 测量最佳的转向位置和转向时间
根据转向位置确定w1和h的大小，根据转向时间确定sleep的时长

3. 停下来的位置不一样和转向位置是不是不一样，
停下来的位置可能离指示牌较近
'''
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

# 定义与训练时相同的 CNN 模型

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*4*4, 100),
            nn.ReLU(),
            nn.Linear(100, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32*4*4)
        x = self.fc(x)
        return x
    
# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model.load_state_dict(torch.load('four.pth', map_location=torch.device('cpu')))
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
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为 HSV 格式以检测蓝色区域
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 蓝色的 HSV 值范围（可以根据实际情况调整）
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    
    # 创建蓝色掩码
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # 寻找轮廓来确定蓝色区域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 设置最小尺寸过滤噪声
        if w > 30 and h > 30:
            # 画出边界框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 提取蓝色区域并预处理为模型输入格式
            sign_img = frame[y:y+h, x:x+w]
            sign_img = cv2.cvtColor(sign_img, cv2.COLOR_BGR2RGB)
            sign_img = Image.fromarray(sign_img)
            sign_img = transform(sign_img).unsqueeze(0).to(device)  # 增加批次维度

            # 将图像输入模型并预测类别
            with torch.no_grad():
                outputs = model(sign_img)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probabilities, 1)
                predicted_class = class_labels[predicted.item()]
                confidence = probabilities[0][predicted.item()].item()

            # 显示预测结果
            cv2.putText(frame, f"{predicted_class}: {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 显示处理后的帧
    cv2.imshow('Traffic Sign Recognition', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

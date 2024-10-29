import os
import random
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import multiprocessing
import math

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
num_epochs = 8
batch_size = 100
lr = 0.001

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),                 # 调整图像大小为28x28
    transforms.Grayscale(num_output_channels=1), # 转换为灰度图像
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道的归一化
])

# 定义 CNN 模型
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
            nn.Linear(100, num_classes)  # 输出类别数量，这里是4
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32 * 4 * 4)
        x = self.fc(x)
        return x

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train() # 打开梯度
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device) # 输入和模型在同一设备中
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward() # 反向传播
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    # 保存训练好的模型
    torch.save(model.state_dict(), 'traffic_sign_model.pth')

# 测试函数
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): # 关闭梯度
        for images, labels in test_loader: 
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# class MyDataset(Dataset):
#     def __init__(self, root):
#         super(MyDataset).__init__()
#         self.root = root
#         self.image, self.label = getImage(root)

#     def __len__(self):
#         return left + pause + right + straight
    
#     def __getitem__(self, index):
#         return image[index], label[index]
    
#     def transform(self, image):
#         transform = transforms.Compose([
#         transforms.Resize((28, 28)),                 # 调整图像大小为28x28
#         transforms.Grayscale(num_output_channels=1), # 转换为灰度图像
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道的归一化
#         ])
#         return transform(image)

# 主函数
def main():
    print(f'Using {device} for training')

    # 加载数据，假设四类图像存储在 data/train 和 data/test 的子文件夹中
    train_dataset = ImageFolder('data/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2) # batch_size:批量处理

    test_dataset = ImageFolder('data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 确定类别数量，应为4类
    num_classes = len(train_dataset.classes)  # 自动识别类别数量
    model = CNN(num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # 测试模型
    test_model(model, test_loader, device)

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows 系统中使用多线程
    main()

# 数据，模型，训练流程
# 1. 数据
# Dataset类 分开训练集与测试集
# 2. 模型
# Model类
# 3. 训练
# 超参config = args，optimizer、loss：反向传播
# train() test()

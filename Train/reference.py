import os
import random
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader
import multiprocessing
import math

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
num_epochs = 20
batch_size = 100
lr = 0.001

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)), # 调整图像大小为28x28
    transforms.Grayscale(num_output_channels=1), # 转换为灰度图像
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) # 修改为单通道的归一化
])

# CNN模型定义
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.Conv2d(16,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(16,32,5),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Dropout(0.3)
        )
        self.fc=nn.Sequential(
            nn.Linear(32*4*4,100),
            nn.ReLU(),
            nn.Linear(100,4)
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
        x=x.view(-1,32*4*4)
        x = self.fc(x)
        return x

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    torch.save(model.state_dict(), 'cnn_model.pth')

def test_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

def main():
    print(f'Using {device} for training')

    # 加载数据
    train_dataset = ImageFolder('data/augmented_train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = ImageFolder('data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化模型
    num_classes = len(train_dataset.classes)
    model = CNN(num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # 测试模型
    test_model(model, test_loader, device)

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 对于Windows系统，这行很重要
    main()
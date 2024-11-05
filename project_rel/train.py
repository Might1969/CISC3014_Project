import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pyarrow import duration
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm
from nn_module import SVHNClassifier
import yaml

# 读取超参数配置文件
def load_config(config_path='result/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 加载配置
config = load_config('result/config.yaml')

# 应用配置中的超参数
learning_rate = config['learning_rate']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
optimizer_type = config['optimizer']
augmentation_params = config['augmentation']
max_rotation = augmentation_params['max_rotation']
min_crop = augmentation_params['min_crop']
max_aspect_ratio = augmentation_params['max_aspect_ratio']

# 3. 训练与评估函数
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10,device='cuda'):
    model.to(device)
    train_losses = []  # 用于保存每个 epoch 的训练损失
    test_losses = []  # 用于保存每个 epoch 的测试损失
    duration = 0
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0

        start_time = time.time()
        # 使用 tqdm 显示训练进度
        for images, labels in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失

            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()  # 累加损失

        end_time = time.time()
        duration += end_time - start_time


        # 计算训练集损失的平均值
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # 计算测试集损失
        model.eval()  # 设置模型为评估模式
        running_test_loss = 0.0

        with torch.no_grad():
            # 使用 tqdm 显示测试进度
            for images, labels in tqdm(test_loader, desc=f'Evaluating Epoch {epoch + 1}/{num_epochs}', unit='batch'):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_test_loss += loss.item()

        # 计算测试集损失的平均值
        test_loss = running_test_loss / len(test_loader)
        test_losses.append(test_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    with open('result/training_log.txt', 'w') as f:
        f.write(f'Train time: {duration:.4f}\n')
    # 绘制损失值与 epoch 的关系图
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', marker='o')
    plt.title('Train and Test Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.savefig('result/loss_plot.png')
    plt.show()
    plt.close()

train_transforms = A.Compose([
            A.RandomResizedCrop(height=32, width=32, scale=(min_crop, 1.0),
                                ratio=(1 / max_aspect_ratio, max_aspect_ratio)),  # 添加 ratio 参数
            A.Rotate(limit=max_rotation),
            A.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
            ToTensorV2()
        ])

test_transforms = A.Compose([
    A.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
    ToTensorV2(),
])

from torchvision import datasets

train_dataset = datasets.SVHN(
    root='./data',
    split='train',
    transform=lambda img: train_transforms(image=np.array(img))['image'],
    download=False
)

test_dataset = datasets.SVHN(
    root='./data',
    split='test',
    transform=lambda img: test_transforms(image=np.array(img))['image'],
    download=False
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = SVHNClassifier()
criterion = nn.CrossEntropyLoss()

if optimizer_type == 'adam': optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if optimizer_type == 'sgd': optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
if optimizer_type == 'rmsprop': optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

train_model(model, train_loader,test_loader, criterion, optimizer, num_epochs=num_epochs)

# 保存模型的状态字典
torch.save(model.state_dict(), 'result/svhn_classifier.pth')


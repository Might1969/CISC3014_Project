import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import gradio as gr
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

def evaluate_model(model, test_loader, class_names, device='cuda'):
    model.to(device)
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    all_labels = []
    all_outputs = []
    duration = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            start_time = time.time()

            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)  # 将输出转换为概率
            _, predicted = torch.max(probabilities, 1)  # 获取预测类别

            end_time = time.time()
            duration += end_time - start_time

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.append(labels.cpu().numpy())
            all_outputs.append(probabilities.cpu().numpy())  # 保存概率而不是 logits

    avg_predict_speed = total / duration
    accuracy = correct / total
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)

    # 计算AUC (macro和micro)
    roc_auc_macro = roc_auc_score(all_labels, all_outputs, multi_class='ovr', average='macro')
    roc_auc_micro = roc_auc_score(all_labels, all_outputs, multi_class='ovr', average='micro')

    with open('result/training_log.txt', 'a') as f:
        f.write(f'Average predict speed: {avg_predict_speed:.4f}/second\n')
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'ROC AUC (Macro): {roc_auc_macro:.4f}\n')
        f.write(f'ROC AUC (Micro): {roc_auc_micro:.4f}')
        # 生成混淆矩阵并绘制
    plot_confusion_matrix(all_labels, np.argmax(all_outputs, axis=1), class_names)

    # 生成每个类别的 ROC 曲线
    plot_roc_curves(all_labels, all_outputs, class_names)

    # 生成每个类别的准确率条形图
    plot_classwise_accuracy(all_labels, np.argmax(all_outputs, axis=1), class_names)

    return accuracy, roc_auc_macro, roc_auc_micro

def plot_confusion_matrix(labels, predictions, class_names):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')

    plt.savefig('result/confusion_matrix.png')
    plt.show()
    plt.close()

def plot_roc_curves(labels, outputs, class_names):
    plt.figure(figsize=(10, 7))
    n_classes = len(class_names)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels == i, outputs[:, i])  # 计算FPR和TPR
        roc_auc = roc_auc_score(labels == i, outputs[:, i])  # 计算每个类别的ROC AUC
        plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Each Class')
    plt.legend(loc='lower right')

    plt.savefig('result/roc_curves.png')
    plt.show()
    plt.close()

def plot_classwise_accuracy(labels, predictions, class_names):
    cm = confusion_matrix(labels, predictions)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)  # 计算每个类别的准确率
    plt.figure(figsize=(10, 5))
    plt.bar(class_names, class_accuracy, color='skyblue')
    plt.ylabel('Accuracy')
    plt.title('Class-wise Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    plt.savefig('result/classwise_accuracy.png')
    plt.show()
    plt.close()

# 定义测试集的预处理策略（仅归一化）
test_transforms = A.Compose([
    A.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
    ToTensorV2(),
])
test_dataset = datasets.SVHN(
    root='./data',
    split='test',
    transform=lambda img: test_transforms(image=np.array(img))['image'],
    download=False
)

torch.set_num_threads(4)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建模型实例
model = SVHNClassifier()

# 加载保存的模型权重
model.load_state_dict(torch.load('result/svhn_classifier.pth', weights_only=True))

class_names = [str(i) for i in range(10)]  # SVHN 类别 0-9
accuracy, roc_auc_macro, roc_auc_micro = evaluate_model(model, test_loader, class_names)

'''
# 展示图像与标签
def imshow(img):
    img = img / 2 + 0.5  # 去标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 获取测试集中的部分图像与标签
dataiter = iter(test_loader)
images, labels = next(dataiter)

# 显示部分图像（这里展示前4张）
for i in range(0,100):  # 可以修改4为你想展示的图片数量


    if labels[i].item() == 9:
        plt.figure(figsize=(2, 2))
        imshow(images[i])
        print(f'Label: {labels[i].item()}')  # 输出对应标签

print(set(labels.numpy()))  # 打印标签的唯一值
'''
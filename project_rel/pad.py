import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from albumentations.pytorch import ToTensorV2
import albumentations as A
import yaml
from pad_nn_module import SVHNClassifier

# 创建模型实例
model = SVHNClassifier()
# 加载保存的模型权重
model.load_state_dict(torch.load('pad/svhn_classifier.pth', weights_only=True))

# 预处理图像
def preprocess_image(image):
    image = image['composite']
    augmentations = A.Compose([
        A.Resize(32, 32),  # 将绘制的图像调整为与训练图像相同的尺寸
        A.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
        ToTensorV2()
    ])
    image = augmentations(image=image)["image"]
    return image.unsqueeze(0)  # 添加batch维度

# 定义模型推理
def predict_digit(image):
    model.eval()  # 确保模型处于评估模式
    image = preprocess_image(image)
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
    return {str(i): probabilities[i] for i in range(10)}

# 创建Gradio界面
iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(crop_size=(256,256), type='numpy', image_mode='RGB', brush=gr.Brush()),
    outputs=gr.Label(num_top_classes=10),  # 展示每个数字的概率
    live=False,  # 实时更新
    title="SVHN Interactive Number Recognition"
)

# 启动界面
iface.launch(server_name="0.0.0.0",inbrowser=True)
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.nn import functional as F
import Transform as transforms
from HC_latest.model import embed_net
from model import embed_net, visible_net_resnet, thermal_net_resnet, base_resnete_part2, base_resnete_part3, \
    base_resnete_share
import cv2
low_dim = 512
n_class = 395
drop = 0
arch = 'resnet50'


def loadingPic():
    # 从测试集中读取一张图片，并显示出来
    img_path = './database/SYSU-MM01/cam1/0001/0001.jpg'
    img = Image.open(img_path)
    imgarray = np.array(img) / 255.0

    plt.figure(figsize=(8, 8))
    plt.imshow(imgarray)
    plt.axis('off')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.RectScale(288, 144),
        transforms.ToTensor(),
        normalize,
    ])
    input_img = transform_test(img).unsqueeze(0)
    return input_img


class GradCAM(nn.Module):
    def __init__(self):
        super(GradCAM, self).__init__()
        self.visible_net = visible_net_resnet(arch=arch)
        self.thermal_net = thermal_net_resnet(arch=arch)
        self.basenet_share = base_resnete_share(arch=arch)
        self.base_resnete_part2 = base_resnete_part2(arch=arch)
        self.base_resnete_part3 = base_resnete_part3(arch=arch)
        self.feat = 0
        # 获取模型的特征提取层
        # base_share
        self.feature1 = nn.Sequential(OrderedDict({
            name: layer for name, layer in model.named_children()
            if name not in ['bn', 'classifier', 'base_resnete_part2', 'base_resnete_part3', 'avgpool', 'feature1',
                            'feature2', 'feature3', 'feature4', 'feature5', 'classifier1', 'classifier2', 'classifier3',
                            'classifier4', 'classifier5', 'visible_net', 'thermal_net']
        }))
        # thermal
        self.feature2 = nn.Sequential(OrderedDict({
            name: layer for name, layer in model.named_children()
            if name not in ['bn', 'classifier', 'base_resnete_part2', 'base_resnete_part3', 'avgpool', 'feature1',
                            'feature2', 'feature3', 'feature4', 'feature5', 'classifier1', 'classifier2', 'classifier3',
                            'classifier4', 'classifier5', 'base_resnete_part', 'visible_net', 'layer3', 'layer4']
        }))
        # visible
        self.feature3 = nn.Sequential(OrderedDict({

            name: layer for name, layer in model.named_children()
            if name not in ['bn', 'classifier', 'base_resnete_part2', 'base_resnete_part3', 'avgpool', 'feature1',
                            'feature2', 'feature3', 'feature4', 'feature5', 'classifier1', 'classifier2', 'classifier3',
                            'classifier4', 'classifier5', 'base_resnete_part', 'thermal_net', 'layer3', 'layer4']
        }))
        # 获取模型最后的平均池化层
        self.avgpool = model.avgpool
        # 获取模型的输出层
        self.classifier = nn.Sequential(OrderedDict([
            ('bn', model.bn),
            ('classifier', model.classifier)
        ]))
        # 生成梯度占位符
        self.gradients = None

    # 获取梯度的钩子函数
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.thermal_net(x)
        x = self.basenet_share(x)
        feat = x.view(x.size(0), -1)
        self.feat =feat
        # 注册钩子
        h = x.register_hook(self.activations_hook)
        # 对卷积后的输出使用平均池化
        # x = self.avgpool(x)
        # x = x.view((1, -1))
        x = self.classifier(feat)
        return x

    # 获取梯度的方法
    def get_activations_gradient(self):
        return self.gradients

    # 获取卷积层输出的方法
    def get_activations(self, x):
        return self.feat


# 获取热力图
def get_heatmap(model, img):
    model.eval()
    img_pre = model(img)
    pre_class = torch.argmax(img_pre[0, :], dim=-1).item()
    # # 获取预测最高的类别
    # pre_class = torch.argmax(img_pre, dim=-1).item()
    # 获取相对于模型参数的输出梯度
    # loss = pre_class.mean()
    # loss.backward()
    img_pre[0, pre_class].backward()
    # 获取模型的梯度
    gradients = model.get_activations_gradient()
    # 计算梯度相应通道的均值
    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # 获取图像在相应卷积层输出的卷积特征
    activations = model.get_activations(loadingPic()).detach()
    print(activations.shape)
    # 每个通道乘以相应的梯度均值
    for i in range(len(mean_gradients)):
        activations[:, i, :, :] *= mean_gradients[i]
    # 计算所有通道的均值输出得到热力图
    heatmap = torch.mean(activations, dim=1).squeeze()
    # 使用Relu函数作用于热力图
    heatmap = F.relu(heatmap)
    # 对热力图进行标准化
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap


# 合并热力图和原题，并显示结果
def merge_heatmap_image(heatmap, image_path):
    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    grad_cam_img = heatmap * 0.4 + img
    grad_cam_img = grad_cam_img / grad_cam_img.max()
    # 可视化图像
    b,g,r = cv2.split(grad_cam_img)
    grad_cam_img = cv2.merge([r,g,b])

    plt.figure(figsize=(8,8))
    plt.imshow(grad_cam_img)
    plt.axis('off')
    plt.show()


model = embed_net(low_dim, n_class, drop, arch)
model.to(torch.device('cpu'))

cam = GradCAM()
# 获取热力图
heatmap = get_heatmap(cam, loadingPic())
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
img_path = './database/SYSU-MM01/cam1/0001/0001.jpg'
merge_heatmap_image(heatmap, img_path)

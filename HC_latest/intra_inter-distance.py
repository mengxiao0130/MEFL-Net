import torchvision.io
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
from model import embed_net
import matplotlib.pyplot as plt
import Transform as transforms
import torch
import cv2

low_dim = 512
n_class = 395
drop = 0
arch = 'resnet50'

# 从测试集中读取一张图片，并显示出来
img_path = './database/SYSU-MM01/cam1/0013/0005.jpg'
img = Image.open(img_path)
imgarray = np.array(img) / 255.0

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.RectScale(288, 144),
    transforms.ToTensor(),
    normalize,
])

input_img = transform_test(img).unsqueeze(0)

model_base = embed_net(low_dim, n_class, drop, arch)
checkpoint = torch.load(
    './save_model/sysu_id_bn_relu_drop_0.0_lr_1.0e-02_dim_512_whc_0.5_thd_0_pimg_8_ds_cos_md_all_resnet50_best.t')
model_base.load_state_dict(checkpoint['net'], strict=False)
start_epoch = checkpoint['epoch']

# 将所有层的参数 requires_grad 设置为 True
for param in model_base.parameters():
    param.requires_grad = True

# 前向传播
output = model_base(input_img, input_img)

# 计算损失函数
loss = torch.mean(output)

# 反向传播
loss.backward()

# 定义一个字典来保存每一层的梯度
gradients = {}


# 定义钩子函数来获取梯度
def get_gradient(module, grad_input, grad_output):
    # 保存梯度
    gradients[module] = grad_input[0]


# 遍历模型的所有模块，并为每一层注册钩子函数
for name, module in model_base.named_modules():
    module.register_backward_hook(get_gradient)

# # 清除梯度缓存
# optimizer.zero_grad()

# 执行其他训练步骤...

# 打印每一层的梯度
for module, gradient in gradients.items():
    print(module, gradient)

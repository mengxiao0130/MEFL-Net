import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import Transform as transforms
from HC_latest.model import embed_net

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
    plt.show()
    return input_img


model = embed_net(low_dim, n_class, drop, arch)
model.to(torch.device('cpu'))

# 定义钩子函数，获取指定层名称的特征
activation = {}  # 保存获取的输出




def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


model.eval()
# # 获取特征
model.basenet_share.base.layer4.register_forward_hook((get_activation('bn')))
# model.bn.register_forward_hook(get_activation('bn')) #[1,512]
_ = model(loadingPic(), loadingPic(), 2)
bn = activation['bn']  # 结果将保存在activation字典中
print(bn.shape)





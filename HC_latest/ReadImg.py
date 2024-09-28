import os
import cv2
from pytorch_grad_cam import GradCAM
from PIL import Image
import numpy as np
from model import embed_net
import matplotlib.pyplot as plt
import Transform as transforms
import torch


def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


def merge_heatmap_image(heatmap, img, i):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # grad_cam_img = heatmap * 0.4 + img
    grad_cam_img = 0.3 * heatmap + img
    grad_cam_img = grad_cam_img / grad_cam_img.max()
    # 可视化图像
    b, g, r = cv2.split(grad_cam_img)
    grad_cam_img = cv2.merge([r, g, b])

    plt.figure(figsize=(8, 8))
    plt.imshow(grad_cam_img)
    plt.axis('off')
    # plt.show()
    plt.savefig('figs/p2_0/{}.png'.format(i))
    print(i)


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x1, x2):
        return self.model(x1, x2)[0]  # 我这里选择了多个输出的第一个，自己视情况而定




########################################
# 数据集路径
org_img_folder = './database/SYSU-MM01'
# 输出维度
low_dim = 512
# 类总数
n_class = 395
drop = 0
arch = 'resnet50'

# 检索文件
imglist = getFileList(org_img_folder, [], 'jpg')
print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n')
i = 0

for imgpath in imglist:
    # imgname = os.path.splitext(os.path.basename(imgpath))[0]
    # print(imgname)
    # CV2读取图片
    img_demo = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    img = Image.open(imgpath)
    #归一化
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transform变换
    # 维度变换
    # to 张量
    transform_test = transforms.Compose([
        transforms.RectScale(288, 144),
        transforms.ToTensor(),
        normalize,
    ])
    #维度压缩
    input_img = transform_test(img).unsqueeze(0)
    #自己的模型
    model_base = embed_net(low_dim, n_class, drop, arch)
    #加载权重
    checkpoint = torch.load(
        './save_model/sysu_id_bn_relu_drop_0.0_lr_1.0e-02_dim_512_whc_0.5_thd_0_pimg_8_ds_cos_md_all_resnet50_best.t')
    model_base.load_state_dict(checkpoint['net'], strict=False)
    #最好的epoch
    start_epoch = checkpoint['epoch']

    # 对每幅图像执行相关操作
    model = SegmentationModelOutputWrapper(model_base)
    #模型评估
    model_base.eval()
    ######
    # 想看哪一层的特征
    target_layers = [model_base.basenet_share.base.layer4[-1]]
    target_layers = [model_base.basenet_share.CBAM]
    # target_layers = [model_base.base_resnet_part2.base.layer4[-1]]
    target_layers = [model_base.base_resnete_part2.base.layer4[-1]]
#############
    input_tensor = input_img  # Create an input tensor image for your model..
    # model  target输入到gardcam模型
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    targets = None
    # 原图输入到cam
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    # 得到该图的热力图
    grayscale_cam = grayscale_cam[0, :]
    #与原图做加运算
    merge_heatmap_image(grayscale_cam, img_demo, i)
    i = i + 1
    if i == 1001:
        break

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


img_path = './database/SYSU-MM01/cam1/0001/0012.jpg'
img = Image.open(img_path)
# imgarray = np.array(img) / 255.0

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


########################
class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x1, x2):
        return self.model(x1, x2)[3]  # 我这里选择了多个输出的第一个，自己视情况而定

# return x_p2, x_p3, y_p2, y_p3, self.l2norm(feat_bn), self.l2norm(out)
#          0     1    2     3     4                           5
model = SegmentationModelOutputWrapper(model_base)
model_base.eval()
target_layers = [model_base.basenet_share.base.layer4[-1]]
target_layers = [model_base.basenet_share.CBAM]
target_layers = [model_base.base_resnete_part2.base.layer4[-1]]
target_layers = [model_base.base_resnete_part3.base.layer4[-1]]



input_tensor = input_img  # Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)


targets = None



# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]


# 合并热力图和原题，并显示结果
def merge_heatmap_image(heatmap, image_path):
    img = cv2.imread(image_path)
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
    plt.show()


merge_heatmap_image(grayscale_cam, img_path)


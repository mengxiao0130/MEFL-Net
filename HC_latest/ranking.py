import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
import torchvision.transforms as transforms
import torch.utils.data as data
import cv2

parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--data_path', default='database/SYSU-MM01/', type=str, help='./test_data')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
# parser.add_argument('--gall-mode', default='single', type=str, help='single or multi')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
args = parser.parse_args()

data_path = 'database/SYSU-MM01/'
test_mode = [1, 2]  # [2, 1]: VIS to IR; [1, 2]: IR to VIS

#######################################################################

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((288, 144)),
    transforms.ToTensor(),
    normalize,
])

#######################################################################
print('==> Loading data..')

query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode)
nquery = len(query_label)
ngall = len(gall_label)

print("Dataset statistics:")
print("  ------------------------------")
print("  subset   | # ids | # images")
print("  ------------------------------")
print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
print("  ------------------------------")

if not os.path.isdir('./save_ranking/ir-vis'):
    os.makedirs('./save_ranking/ir-vis')
#######################################################################
# Evaluate
# matplotlib.use('TkAgg')
max_True = 0
for j in range(4000):
    if j % 10 == 0:
        print(j)
        result = scipy.io.loadmat('ir_vis.mat')
        query_feature = torch.FloatTensor(result['query_f'])
        query_cam = result['query_cam'][0]
        query_label = result['query_label'][0]
        gallery_feature = torch.FloatTensor(result['gallery_f'])
        gallery_cam = result['gallery_cam'][0]
        gallery_label = result['gallery_label'][0]

        query_feature = query_feature.cuda(0)
        gallery_feature = gallery_feature.cuda(0)


        #####################################################################
        # Show result
        def imshow(path, title=None):
            """Imshow for Tensor."""
            im = plt.imread(path)
            im = cv2.resize(im, (args.img_w, args.img_h))
            plt.imshow(im)
            if title is not None:
                plt.title(title)
            plt.pause(0.001)  # pause a bit so that plots are updated


        #######################################################################
        # sort the images
        def sort_img(qf, ql, qc, gf, gl, gc):
            query = qf.view(-1, 1)
            score = torch.mm(gf, query)
            score = score.squeeze(1).cpu()
            score = score.numpy()
            # predict index
            index = np.argsort(score)  # from small to large
            index = index[::-1]
            # good index
            query_index = np.argwhere(gl == ql)
            # same camera
            camera_index = np.argwhere(gc == qc)

            junk_index1 = np.argwhere(gl == -1)
            junk_index2 = np.intersect1d(query_index, camera_index)
            junk_index = np.append(junk_index2, junk_index1)

            mask = np.in1d(index, junk_index, invert=True)
            index = index[mask]
            return index


        i = j
        index = sort_img(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)

        query_path = query_img[i]
        query_label = query_label[i]
        print(query_path)
        print('Top 10 images are as follow:')
        # Visualize Ranking Result
        # Graphical User Interface is needed
        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        imshow(query_path, 'query')
        for i in range(10):
            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            img_path = gall_img[index[i]]
            label = gallery_label[index[i]]
            imshow(img_path)
            if int(label) == int(query_label):
                ax.set_title('%d' % (i + 1), color='green')
                print(img_path, 'True')
                max_True = max_True + 1
            else:
                ax.set_title('%d' % (i + 1), color='red')
                print(img_path, 'False')
        print("正确个数--------{}".format(max_True))
        max_True = 0
        fig.savefig("./save_ranking/ir-vis/show" + str(j) + ".jpg")

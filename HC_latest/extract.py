from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import TestData
from data_manager import *
from model import embed_net
from utils import *
import scipy.io

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r',
                    default='sysu_id_bn_relu_drop_0.0_lr_1.0e-02_dim_512_whc_0.5_thd_0_pimg_8_ds_cos_md_all_resnet50_best.t',
                    type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=512, type=int, metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=32, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='id', type=str, metavar='m', help='method type')
parser.add_argument('--drop', default=0.0, type=float, metavar='drop', help='dropout ratio')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--per_img', default=8, type=int, help='number of samples of an id in every batch')
parser.add_argument('--w_hc', default=0.5, type=float, help='weight of Hetero-Center Loss')
parser.add_argument('--thd', default=0, type=float, help='threshold of Hetero-Center Loss')
parser.add_argument('--epochs', default=80, type=int, help='weight of Hetero-Center Loss')
parser.add_argument('--dist-type', default='cos', type=str, help='type of distance')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--gall-mode', default='single', type=str, help='single or multi')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
if dataset == 'sysu':
    data_path = 'database/SYSU-MM01/'
    n_class = 395
    test_mode = [1, 2]  # gallery = 1 querry = 2
elif dataset == 'regdb':
    data_path = 'database/RegDB/'
    n_class = 206
    test_mode = [1, 2]  # [2, 1]: VIS to IR; [1, 2]: IR to VIS

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
pool_dim = 2048

print('==> Building model..')
net = embed_net(args.low_dim, n_class, drop=args.drop, arch=args.arch)
net.to(device)
cudnn.benchmark = True

checkpoint_path = args.model_path  # save_model

if args.method == 'id':
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    #  transforms.RectScale(args.img_h, args.img_w),
    transforms.ToTensor(),
    normalize,
])

###############################
end = time.time()

feature_dim = args.low_dim


def extract_gall_feat(gall_loader):
    net.eval()
    print('Extracting Gallery Feature...')
    # print('Extracting Gallery Feature...', file=test_log_file)
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 5 * feature_dim))
    gall_feat_pool = np.zeros((ngall, 5 * pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            pool_feat, feat = net(input, input, test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_pool[ptr:ptr + batch_num, :] = pool_feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    # print('Extracting Time:\t {:.3f}'.format(time.time() - start), file=test_log_file)
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat, gall_feat_pool


def extract_query_feat(query_loader):
    net.eval()
    print('Extracting Query Feature...')
    # print('Extracting Query Feature...', file=test_log_file)
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 5 * feature_dim))
    query_feat_pool = np.zeros((nquery, 5 * pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            pool_feat, feat = net(input, input, test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_pool[ptr:ptr + batch_num, :] = pool_feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    # print('Extracting Time:\t {:.3f}'.format(time.time() - start), file=test_log_file)
    return query_feat, query_feat_pool


def process_llcm(img_dir, mode=1):
    if mode == 1:
        input_data_path = os.path.join(data_path, 'idx/test_vis.txt')
    elif mode == 2:
        input_data_path = os.path.join(data_path, 'idx/test_nir.txt')

    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        file_cam = [int(s.split('c0')[1][0]) for s in data_file_list]

    return file_image, np.array(file_label), np.array(file_cam)


if dataset == 'sysu':
    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        # loading model
        if os.path.isfile(model_path):
            print("loading success")
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode)
    print("query-cam")
    print(query_cam)
    nquery = len(query_label)
    ngall = len(gall_label)

    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=0)

    trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=0)

    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat_pool, query_feat_fc = extract_query_feat(query_loader)
    gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader)

    result = {'gallery_f': gall_feat_fc, 'gallery_label': gall_label, 'gallery_cam': gall_cam, 'query_f': query_feat_fc,
              'query_label': query_label, 'query_cam': query_cam}

    scipy.io.savemat('ir_vis.mat', result)

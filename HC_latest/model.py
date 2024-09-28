import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.nn import functional as F
from CBAM_ATT import CBAMBlock


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# kaiming初始化
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


# 分类器初始化
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
# 特征提取器
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]

        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x


# 分类器
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]

        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x

    # Define the ResNet18-based Model


class visible_net_resnet(nn.Module):
    def __init__(self, arch='resnet18'):
        super(visible_net_resnet, self).__init__()
        if arch == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch == 'resnet50':
            model_ft = models.resnet50(pretrained=True)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.visible = model_ft
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 前向传播：卷积块+BN+RELU+最大池化+4个层
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        x = self.visible.layer2(x)
        return x


class thermal_net_resnet(nn.Module):
    def __init__(self, arch='resnet18'):
        super(thermal_net_resnet, self).__init__()
        if arch == 'resnet18':
            model_ft = models.resnet18(pretrained=True)
        elif arch == 'resnet50':
            model_ft = models.resnet50(pretrained=True)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.thermal = model_ft
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        x = self.thermal.layer2(x)
        return x




# global+6part
# part嵌入
class embed_net(nn.Module):
    def __init__(self, low_dim, class_num, drop=0.5, arch='resnet50', gm_pool='on'):
        super(embed_net, self).__init__()
        if arch == 'resnet18':
            self.visible_net = visible_net_resnet(arch=arch)
            self.thermal_net = thermal_net_resnet(arch=arch)
            self.basenet_share = base_resnete_share(arch=arch)
            self.base_resnete_part2 = base_resnete_part2(arch=arch)
            self.base_resnete_part3 = base_resnete_part3(arch=arch)
            pool_dim = 512
        elif arch == 'resnet50':
            self.visible_net = visible_net_resnet(arch=arch)
            self.thermal_net = thermal_net_resnet(arch=arch)
            self.basenet_share = base_resnete_share(arch=arch)
            self.base_resnete_part2 = base_resnete_part2(arch=arch)
            self.base_resnete_part3 = base_resnete_part3(arch=arch)
            pool_dim = 2048

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.gm_pool = gm_pool

        # 特征由FeatureBlock提取
        # 分类器是ClassBlock
        self.feature1 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature2 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature3 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature4 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature5 = FeatureBlock(pool_dim, low_dim, dropout=drop)

        self.classifier1 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier2 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier3 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier4 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier5 = ClassBlock(low_dim, class_num, dropout=drop)
        self.conv0 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv4 = nn.Conv2d(1024, 512, kernel_size=1)
        self.classifier = ClassBlock(low_dim, class_num, dropout=drop)
        self.bn = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.l2norm = Normalize(2)

    def forward(self, x1, x2, modal=0):
        # def forward(self, x2, modal=2):
        if modal == 0:
            # modal为0：上分支为可见光模态，下分支为红外模态
            # x1是可见光图像的特征映射
            x1 = self.visible_net(x1)
            x2 = self.thermal_net(x2)
            x = torch.cat((x1, x2), 0)

            global_feature = self.basenet_share(x)
            part2 = self.base_resnete_part2(x)
            part3 = self.base_resnete_part3(x)
            # chunk方法可以对张量分块，返回一个张量列表
            # 沿2轴分为6块
            x_part2 = part2.chunk(2, 2)
            x_part3 = part3.chunk(3, 2)
            # 将特征分为6个条纹
            # pytorch contiguous一般与transpose，permute,view搭配使用：使用transpose或permute进行维度变换后，调用contiguous,然后方可使用view对维度进行变形
            x_0 = x_part2[0].contiguous().view(x_part2[0].size(0), -1)
            x_1 = x_part2[1].contiguous().view(x_part2[1].size(0), -1)

            x_2 = x_part3[0].contiguous().view(x_part3[0].size(0), -1)
            x_3 = x_part3[1].contiguous().view(x_part3[1].size(0), -1)
            x_4 = x_part3[2].contiguous().view(x_part3[2].size(0), -1)
        elif modal == 1:
            # modal为1表示是可见模态
            x = self.visible_net(x1)
            global_feature = self.basenet_share(x)
            part2 = self.base_resnete_part2(x)
            part3 = self.base_resnete_part3(x)
            # chunk方法可以对张量分块，返回一个张量列表
            # 沿2轴分为6块
            x_part2 = part2.chunk(2, 2)
            x_part3 = part3.chunk(3, 2)
            # 将特征分为6个条纹
            # pytorch contiguous一般与transpose，permute,view搭配使用：使用transpose或permute进行维度变换后，调用contiguous,然后方可使用view对维度进行变形
            x_0 = x_part2[0].contiguous().view(x_part2[0].size(0), -1)
            x_1 = x_part2[1].contiguous().view(x_part2[1].size(0), -1)

            x_2 = x_part3[0].contiguous().view(x_part3[0].size(0), -1)
            x_3 = x_part3[1].contiguous().view(x_part3[1].size(0), -1)
            x_4 = x_part3[2].contiguous().view(x_part3[2].size(0), -1)
        elif modal == 2:
            # modal为2表示是红外模态
            x = self.thermal_net(x2)

            global_feature = self.basenet_share(x)
            part2 = self.base_resnete_part2(x)
            part3 = self.base_resnete_part3(x)
            # chunk方法可以对张量分块，返回一个张量列表
            # 沿2轴分为6块
            x_part2 = part2.chunk(2, 2)
            x_part3 = part3.chunk(3, 2)
            # 将特征分为6个条纹
            # pytorch contiguous一般与transpose，permute,view搭配使用：使用transpose或permute进行维度变换后，调用contiguous,然后方可使用view对维度进行变形
            x_0 = x_part2[0].contiguous().view(x_part2[0].size(0), -1)
            x_1 = x_part2[1].contiguous().view(x_part2[1].size(0), -1)

            x_2 = x_part3[0].contiguous().view(x_part3[0].size(0), -1)
            x_3 = x_part3[1].contiguous().view(x_part3[1].size(0), -1)
            x_4 = x_part3[2].contiguous().view(x_part3[2].size(0), -1)
        ##############

        # global

        feat = global_feature.view(global_feature.size(0), -1)

        feat_bn = self.bn(feat)
        out = self.classifier(feat_bn)
        ####################

        # part
        # BN
        y_0 = self.feature1(x_0)
        y_1 = self.feature2(x_1)

        y_2 = self.feature2(x_2)
        y_3 = self.feature2(x_3)
        y_4 = self.feature2(x_4)

        y_0 = y_0.unsqueeze(2).unsqueeze(3)
        y_1 = y_1.unsqueeze(2).unsqueeze(3)
        y_2 = y_2.unsqueeze(2).unsqueeze(3)
        y_3 = y_3.unsqueeze(2).unsqueeze(3)
        y_4 = y_4.unsqueeze(2).unsqueeze(3)

        y01 = y_0 + y_1

        y23 = y_2 + y_3

        y34 = y_3 + y_4

        y24 = y_2 + y_4

        y_0 = torch.cat((y_0, y01), dim=1)
        y_1 = torch.cat((y_1, y01), dim=1)
        y_2 = torch.cat((y_2, y34), dim=1)
        y_3 = torch.cat((y_3, y24), dim=1)
        y_4 = torch.cat((y_4, y23), dim=1)
        y_0 = self.conv0(y_0)
        y_1 = self.conv1(y_1)
        y_2 = self.conv2(y_2)
        y_3 = self.conv3(y_3)
        y_4 = self.conv4(y_4)
        y_0 = y_0.squeeze(2).squeeze(2)
        y_1 = y_1.squeeze(2).squeeze(2)
        y_2 = y_2.squeeze(2).squeeze(2)
        y_3 = y_3.squeeze(2).squeeze(2)
        y_4 = y_4.squeeze(2).squeeze(2)


        # 对特征y进行分类

        out_0 = self.classifier1(y_0)

        out_1 = self.classifier2(y_1)

        out_2 = self.classifier3(y_2)
        out_3 = self.classifier4(y_3)
        out_4 = self.classifier5(y_4)

        if self.training:
            return (out_0, out_1, out_2, out_3, out_4), (
                self.l2norm(y_0), self.l2norm(y_1), self.l2norm(y_2), self.l2norm(y_3), self.l2norm(y_4)), self.l2norm(
                feat_bn), out
        else:
            x_0 = self.l2norm(x_0)
            x_1 = self.l2norm(x_1)

            x_2 = self.l2norm(x_2)
            x_3 = self.l2norm(x_3)
            x_4 = self.l2norm(x_4)
            x = torch.cat((x_0, x_1, x_2, x_3, x_4), 1)
            x_p2 = torch.cat((x_0, x_1))
            x_p3 = torch.cat((x_2, x_3, x_4))
            y_0 = self.l2norm(y_0)
            y_1 = self.l2norm(y_1)

            y_2 = self.l2norm(y_2)
            y_3 = self.l2norm(y_3)
            y_4 = self.l2norm(y_4)

            y = torch.cat((y_0, y_1, y_2, y_3, y_4), 1)
            y_p2 = torch.cat((y_0, y_1))
            y_p3 = torch.cat((y_2, y_3, y_4))
            return x, y
            # return x_p2, x_p3, y_p2, y_p3, self.l2norm(feat_bn), self.l2norm(out)

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np


def make_layers(cfg, in_channels = 3,batch_norm=False, pad=1, kernel_size=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        elif v == 'R':
            layers += nn.ReLU(inplace=True)
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=kernel_size, padding=pad)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def Class_to_count(cls, label):
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label)
    label = label.squeeze()

    cnt = [0.0]
    for i in range(len(label)-1):
        cnt.append((label[i] + label[i+1]) / 2)
    cnt.append(label[i])
    cnt = torch.tensor(cnt)
    cnt = cnt.type(torch.FloatTensor)

    output_size = cls.size()
    cls = cls.reshape(-1).cpu()
    pre = torch.index_select(cnt, 0, cls.cpu().type(torch.LongTensor))
    pre = pre.reshape(output_size)
    if cls.device.type == 'cuda':
        pre = pre.cuda()
    return pre

def count_merge(c1, c2):
    rt = int(c2.size([-1]/c1.size()[-1]))
    norm = 1/(float(rt)**2)
    res = torch.zeros(c2.size())
    c1, c2, res = c1.cuda(), c2.cuda(), res.cuda()

    for x in range(rt):
        for y in range(rt):
            res[:, :, x::rt, y::rt] = c1 * norm

    if c1.device.type == 'cuda':
        res = res.cpu()
    return res

class MFM_Generator(nn.Module):
    def __init__(self, mfm_idx):
        if mfm_idx == 1:
            self.before_conv = 768
        else:
            self.before_conv = 512
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(self.before_conv, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, fm0, fm1):
        fm0 = self.upsample(fm0)
        fm0 = self.conv_layer1(fm0)
        fm0 = F.pad(fm0, ((fm1.size()[3] - fm0.size()[3]) // 2, int(math.ceil((fm1.size()[3] - fm0.size()[3]) / 2.0)),
                        (fm1.size()[2] - fm0.size()[2]) // 2, int(math.ceil((fm1.size()[2] - fm0.size()[2]) / 2.0))))
        mfm1 = torch.cat([fm1, fm0], dim=1)
        mfm1 = self.conv_layer2(mfm1)
        return mfm1



class Final_Net(nn.Module):
    def __init__(self, class_number, label):
        super(Final_Net, self).__init__()

        self.label = label
        self.conv_layer1 = make_layers([64, 64, 'M'], in_channels=3)
        self.conv_layer2 = make_layers([128, 128, 'M'], in_channels=64)
        self.conv_layer3 = make_layers([256, 256, 256, 'M'], in_channels=128)
        self.conv_layer4 = make_layers([512, 512, 512, 'M'], in_channels=256)
        self.conv_layer5 = make_layers([512, 512, 'M'], in_channels=512)

        self.getMFM1 = MFM_Generator(mfm_idx=1)
        self.getMFM2 = MFM_Generator(mfm_idx=2)

        self.classifier = make_layers(['A', 'R', 512, class_number], in_channels=512, kernel_size=1, pad=0)
        self.divison_mask_generator = make_layers(['A', 'R', 512, 1], in_channels=512, kernel_size=1, pad=0)
        self.dm = dm_generator()

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        fm2 = x
        x = self.conv_layer4(x)
        fm1 = x
        x = self.conv_layer5(x)
        fm0 = x
        x = self.classifier(x)
        feature_map_set_before_merge = {'fm0': fm0, 'fm1': fm1, 'fm2': fm2, 'c0': x}
        weight_and_class = self.generate_weights_and_class(feature_map_set_before_merge)
        res_high = self.get_counting_res(weight_and_class)
        dm = self.dm(res_high)
        return dm

    def generate_weights_and_class(self, feature_map_set_before_merge):
        mfm1 = self.getMFM1(feature_map_set_before_merge['fm0'], feature_map_set_before_merge['fm1'])
        mfm1_weight = self.divison_mask_generator(mfm1)
        mfm1_class = self.classifier(mfm1)

        mfm2 = self.getMFM2(mfm1, feature_map_set_before_merge['fm2'])
        mfm2_weight = self.divison_mask_generator(mfm2)
        mfm2_class = self.classifier(mfm2)

        weight_and_class = {'cls0': feature_map_set_before_merge['c0'], 'cls1': mfm1_class, 'cls2': mfm2_class,
                            'w1': 1 - mfm1_weight, 'w2': 1 - mfm2_weight}
        return weight_and_class

    def get_counting_res(self, weight_and_class):
        c0 = Class_to_count(weight_and_class['cls0'].max(dim=1, keepdim=True)[1], self.label)
        c1 = Class_to_count(weight_and_class['cls1'].max(dim=1, keepdim=True)[1], self.label)
        c2 = Class_to_count(weight_and_class['cls2'].max(dim=1, keepdim=True)[1], self.label)

        cnt_low = count_merge(c0, c1)
        res_low = (1-weight_and_class['w1']) * cnt_low + weight_and_class['w1'] * c1
        cnt_high = count_merge(res_low, c2)
        res_high = (1-weight_and_class['w2']) * cnt_high + weight_and_class['w2'] * c2

        return res_high


class dm_generator(nn.Module):
    def __init__(self):
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = make_layers([64, 256, 256], in_channels=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = make_layers([512, 512, 512, 128, 64, 1], in_channels=256)

    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        return x
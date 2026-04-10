import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.mymodels import Octave  # 八度卷积
from lib.mymodels import ConcatHighConv,ConcatHLConv  # 高频融合
from lib.mymodels import SCF  # 尺度扩张，残差融合
from lib.mymodels import CCM # 特征融合
from lib.pvtv2 import pvt_v2_b2

class two_ConvBnRule(nn.Module):
    def __init__(self, in_chan, out_chan=64):
        super(two_ConvBnRule, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN1 = nn.BatchNorm2d(out_chan)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_chan,
            out_channels=out_chan,
            kernel_size=3,
            padding=1
        )
        self.BN2 = nn.BatchNorm2d(out_chan)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, mid=False):
        feat = self.conv1(x)
        feat = self.BN1(feat)
        feat = self.relu1(feat)
        feat = self.conv2(feat)
        feat = self.BN2(feat)
        feat = self.relu2(feat)
        return feat

class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1, group=1,
                    has_bn=True, has_relu=True, mode='2d'):
        super(ConvBNReLU, self).__init__()
        self.has_bn = has_bn
        self.has_relu = has_relu
        if mode == '2d':
            self.conv = nn.Conv2d(
                    c_in, c_out, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=False, groups=group)
            norm_layer = nn.BatchNorm2d
        elif mode == '1d':
            self.conv = nn.Conv1d(
                    c_in, c_out, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=False, groups=group)
            norm_layer = nn.BatchNorm1d
        if self.has_bn:
            self.bn = norm_layer(c_out)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        # def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=True, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FCCNet(nn.Module):
    def __init__(self, channel=64, pretrained=True):
        super(FCCNet, self).__init__()
        # ---- PVT Backbone ----
        # self.pvt = pvt_v1(pretrained=imagenet_pretrained)
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        if pretrained:
            path = 'pvt_v2_b2.pth'
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
        else:
            pass

        self.rfb4_1 = Octave(512, channel)
        self.rfb3_1 = Octave(320, channel)
        self.rfb2_1 = Octave(128, channel)
        self.high = ConcatHighConv(channel*3,channel)
        self.low = ConcatHighConv(channel*3,channel)
        self.hl_cat = ConcatHLConv(channel*2,channel)
        # boundary and S_g
        self.conv_h = ConvBNReLU(channel,channel,3,stride=1,padding=1)
        self.conv_b = BasicConv2d(channel,1,1)
        # IDR
        self.scf = SCF(channel, channel)
        self.scf2 = SCF(channel, channel)
        self.scf3 = SCF(channel, channel)

        self.conv_f1 = ConvBNReLU(64,channel,1,stride=1,padding=0)
        self.conv_f2 = ConvBNReLU(128,channel,1,stride=1,padding=0)
        self.conv_f3 = ConvBNReLU(320,channel,1,stride=1,padding=0)
        self.conv_f4 = ConvBNReLU(512,channel,1,stride=1,padding=0)

        self.ccm = CCM(channel)
        self.ccm2 = CCM(channel)
        self.ccm3 = CCM(channel)

        # 输出gt_pred
        self.conv_out = nn.Conv2d(channel, 1, 1)
        self.conv_out_e = nn.Conv2d(channel, 1, 1)
        self.conv_out1 = nn.Conv2d(channel, 1, 1)
        self.conv_out2 = nn.Conv2d(channel, 1, 1)
        self.conv_out3 = nn.Conv2d(channel, 1, 1)

    def forward(self, x):
        outs = self.backbone(x)
        x0_rfb = self.conv_f1(outs[0])
        x1_rfb = self.conv_f2(outs[1])
        x2_rfb = self.conv_f3(outs[2])
        x3_rfb = self.conv_f4(outs[3])
        # 八度卷积
        x2_rfb_h,x2_rfb_l,x2_rfb_c = self.rfb2_1(outs[1])  # channel -> 64  stage3对应的八度卷积
        x3_rfb_h,x3_rfb_l,x3_rfb_c = self.rfb3_1(outs[2])  # channel -> 64  stage4对应的八度卷积
        x4_rfb_h,x4_rfb_l,x4_rfb_c = self.rfb4_1(outs[3])  # channel -> 64  stage5对应的八度卷积
        ## FCM
        x_h_cat = self.high(x2_rfb_h,x3_rfb_h,x4_rfb_h)
        x_l_cat = self.low(x2_rfb_l,x3_rfb_l,x4_rfb_l)
        # # 生成boundary
        x_h_b = self.conv_h(x_h_cat) # 3*3
        x_h_b = self.conv_out_e(x_h_b) # 3*3
        edge = torch.sigmoid(x_h_b)
        # S_g
        S_g = self.hl_cat(x_h_cat, x_l_cat)
        S_g = self.conv_h(S_g) # [1,64,44,44]
        S_g_out = self.conv_out(S_g) # [1,1,44,44] 1*1
        p3 = self.scf3(x3_rfb, x2_rfb)
        p2 = self.scf2(x2_rfb, x1_rfb)
        p1 = self.scf(x1_rfb, x0_rfb)
        S_g_guide34 = F.interpolate(S_g, scale_factor=0.5, mode='bilinear')  # 缩小0.5  torch.Size([1, 32, 22, 22])
        lateral_map_g = F.interpolate(S_g_out, scale_factor=8, mode='bilinear')
        S3 = self.ccm3(S_g_guide34, p3, edge)
        S3_out = self.conv_out3(S3)
        lateral_map_3 = F.interpolate(S3_out, scale_factor=16, mode='bilinear')
        S_g_guide23 = F.interpolate(S3, scale_factor=2, mode='bilinear')  # 扩大2倍  (1,44,44)
        S2 = self.ccm2(S_g_guide23, p2, edge)
        S2_out = self.conv_out2(S2)
        lateral_map_2 = F.interpolate(S2_out, scale_factor=8, mode='bilinear')
        S_g_guide12 = F.interpolate(S2, scale_factor=2, mode='bilinear')  # 扩大2倍  (1,88,88)
        S1 = self.ccm(S_g_guide12, p1, edge)
        S1_out = self.conv_out1(S1)
        lateral_map_1 = F.interpolate(S1_out, scale_factor=4, mode='bilinear')
        # edge map
        edge_map = F.interpolate(edge, scale_factor=8, mode='bilinear', align_corners=False) # torch.Size([1, 1, 352, 352])
        return lateral_map_g, lateral_map_3, lateral_map_2,lateral_map_1,edge_map


if __name__ == '__main__':
    print("hello")
    net = FCCNet(channel=32, pretrained=False)
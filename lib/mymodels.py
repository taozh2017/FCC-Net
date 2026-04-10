import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


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


# class RFB_modified(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         self.relu = nn.ReLU(inplace=True)
#         self.branch0 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#             BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
#             BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
#             BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#             BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
#             BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
#             BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
#         )
#         self.branch3 = nn.Sequential(
#             BasicConv2d(in_channel, out_channel, 1),
#             BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
#             BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
#             BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
#         )
#         self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
#         self.conv_res = BasicConv2d(in_channel, out_channel, 1)
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         x3 = self.branch3(x)
#
#         x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
#
#         x = self.relu(x_cat + self.conv_res(x))
#         return x


class FirstOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * in_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, in_channels - int(alpha * in_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        if self.stride ==2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x) # 低频，就像GAP是DCT最低频的特殊情况？长宽缩小一半
        X_h = x
        X_h = self.h2h(X_h)
        X_l = self.h2l(X_h2l)

        return X_h, X_l

class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        # X_l2h = self.upsample(X_l2h)
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]),int(X_h2h.size()[3])), mode='bilinear')
        # print('X_l2h:{}'.format(X_l2h.shape))
        # print('X_h2h:{}'.format(X_h2h.shape))
        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l

class LastOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.l2h = torch.nn.Conv2d(int(alpha * out_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(out_channels - int(alpha * out_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2h = self.h2h(X_h) # 高频组对齐通道
        X_l2h = self.l2h(X_l) # 低频组对齐通道
        # 低频组对齐长宽尺寸
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]), int(X_h2h.size()[3])), mode='bilinear')

        X_h = X_h2h + X_l2h  # 本来的设置：高频低频融合输出
        # return X_h       #都输出

        # return X_h2h  #只输出高频组
        # return X_l2h    #只输出低频组

        return X_h2h, X_l2h,X_h

class Octave(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(Octave, self).__init__()
        # 第一层，将特征分为高频和低频
        self.fir = FirstOctaveConv(in_channels, out_channels, kernel_size)
        # 第二层，低高频输入，低高频输出
        self.mid1 = OctaveConv(in_channels, in_channels, kernel_size)
        self.mid2 = OctaveConv(in_channels, out_channels, kernel_size)
        # 第三层，将低高频汇合后输出
        self.lst = LastOctaveConv(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x0 = x
        x_h, x_l = self.fir(x)                   # (1,64,64,64) ,(1,64,32,32)
        x_hh, x_ll = x_h, x_l,
        # x_1 = x_hh +x_ll
        x_h_1, x_l_1 = self.mid1((x_h, x_l))     # (1,64,64,64) ,(1,64,32,32)
        x_h_2, x_l_2 = self.mid1((x_h_1, x_l_1)) # (1,64,64,64) ,(1,64,32,32)
        x_h_5, x_l_5 = self.mid2((x_h_2, x_l_2)) # (1,32,64,64) ,(1,32,32,32)
        x_ret_h,x_ret_l,x_ret = self.lst((x_h_5, x_l_5)) # (1,64,64,64)
        return x_ret_h,x_ret_l,x_ret # 高频，低频,融合

        # x_l_11 = F.interpolate(x_l_1, (int(x_h_1.size()[2]), int(x_h_1.size()[3])), mode='bilinear')
        # x_ret, x_h_6, x_l_6 = self.lst((x_h_5, x_l_5)) # (1,64,64,64)
        # return x0, x_ret,x_hh, x_ll,x_h_1, x_l_1

        # return x0, x_ret, x_hh, x_ll, x_h_6, x_l_6
        # return x0, x_ret
    # fea_name = ['_before','_after', '_beforeH', '_beforeL', '_afterH', '_afterL', '_afterH0', '_afterL0']

import torch
from torch import nn


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
# # 空间注意力模块SAM
# class SpatialAttention(nn.Module):  # 空间注意力模块
#     def __init__(self, in_channel):
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv2d(2, 1, 7, padding=3)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):  # (64,176,176)
#         # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]
#         MaxPool = torch.max(x, dim=1).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
#         AvgPool = torch.mean(x, dim=1)  # (176,176)
#
#         # 增加维度, 变成 [N, 1, H, W]
#         MaxPool = torch.unsqueeze(MaxPool, dim=1) # (1,176,176)
#         AvgPool = torch.unsqueeze(AvgPool, dim=1)
#
#         # 维度拼接 [N, 2, H, W]   # (2,176,176)
#         x_cat = torch.cat((MaxPool, AvgPool), dim=1)  # 获得特征图
#
#         # 卷积操作得到空间注意力结果
#         x_out = self.conv(x_cat) # (1,176,176)
#         Ms = self.sigmoid(x_out)
#
#         # 与原图通道进行乘积
#         x = Ms * x + x
#
#         return x # (64,176,176)

# 低频concat并融合高频的信息
class ConcatHLConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConcatHLConv, self).__init__()
        # self.conv1 = BasicConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1 = ConvBNReLU(in_channels, out_channels, 3,stride=1, padding=1)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x_h, x_l):
        # Concatenate along the channel dimension
        x_concat = torch.cat((x_h,x_l), dim=1)

        # Apply the second convolution
        x_result = self.conv1(x_concat)

        return x_result

class ConcatHighConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConcatHighConv, self).__init__()
        self.conv1 = BasicConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x2_rfb_h, x3_rfb_h, x4_rfb_h):
        # Concatenate along the channel dimension
        x3_rfb_h = F.interpolate(x3_rfb_h, size=(x2_rfb_h.size(2), x2_rfb_h.size(3)), mode='bilinear')
        x4_rfb_h = F.interpolate(x4_rfb_h, size=(x2_rfb_h.size(2), x2_rfb_h.size(3)), mode='bilinear')
        x_concat = torch.cat((x2_rfb_h, x3_rfb_h, x4_rfb_h), dim=1)

        # Apply the first convolution
        x_conv1 = self.conv1(x_concat)

        # Apply the second convolution
        # x_result = self.conv2(x_conv1)

        return x_conv1

# class ConcatHighConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ConcatHighConv, self).__init__()
#         # self.conv1 = BasicConv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.conv1 = ConvBNReLU(out_channels, out_channels, 3,stride=1, padding=1)
#         self.softmax = nn.Softmax(dim=1)
#         self.out_c = out_channels
#         # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

#     def forward(self, x2_rfb_h, x3_rfb_h, x4_rfb_h):
#         # Concatenate along the channel dimension
#         x3_rfb_h = F.interpolate(x3_rfb_h, size=(x2_rfb_h.size(2), x2_rfb_h.size(3)), mode='bilinear')
#         x4_rfb_h = F.interpolate(x4_rfb_h, size=(x2_rfb_h.size(2), x2_rfb_h.size(3)), mode='bilinear')
#         # x_concat = torch.cat((x2_rfb_h, x3_rfb_h, x4_rfb_h), dim=1)
#         x_concat = torch.cat((x2_rfb_h, x3_rfb_h, x4_rfb_h), dim=1)
#         x_concat = self.softmax(x_concat)
#         x_concat_2, x_concat_3,x_concat_4 = x_concat[:, 0:self.out_c, :, :], x_concat[:, self.out_c:2*self.out_c, :, :], x_concat[:, 2*self.out_c:3*self.out_c, :, :]
#         # print("x_concat_2.shape,x_concat_3.shape,x_concat_4.shape",x_concat_2.shape,x_concat_3.shape,x_concat_4.shape)
#         merge_feature = x2_rfb_h * x_concat_2 + x3_rfb_h * x_concat_3+x4_rfb_h*x_concat_4
#         # print("merge_feature.shape",merge_feature.shape)
#         # Apply the first convolution
#         x_conv1 = self.conv1(merge_feature)

#         # Apply the second convolution
#         # x_result = self.conv2(x_conv1)

#         return x_conv1

class SCF(nn.Module):
    def __init__(self, enc_channels, channels=64,dilation=[1,3,5,7]):
        super(SCF, self).__init__()

        # 每个输入通道扩张为原来的4倍
        self.scale_expand = ConvBNReLU(enc_channels, channels * 4, kernel_size=1,stride=1,padding=0)

        self.hidden_conv1 = BasicConv2d(enc_channels, channels, kernel_size=3, padding=dilation[0], dilation=dilation[0])
        # self.hidden_conv1 = ConvBNReLU(enc_channels, channels, kernel_size=3, stride=1,padding=dilation[0], dilation=dilation[0])
        # self.hidden_conv2 = ConvBNReLU(enc_channels, channels, kernel_size=3, stride=1,padding=dilation[1], dilation=dilation[1])
        self.hidden_conv2 = BasicConv2d(enc_channels, channels, kernel_size=3, padding=dilation[1], dilation=dilation[1])
        # 仿照resnet
        self.hidden_conv22 = BasicConv2d(enc_channels*2, channels*2, kernel_size=3, padding=dilation[1], dilation=dilation[1])
        # self.hidden_conv3 = ConvBNReLU(enc_channels, channels, kernel_size=3, stride=1,padding=dilation[2], dilation=dilation[2])
        self.hidden_conv3 = BasicConv2d(enc_channels, channels, kernel_size=3, padding=dilation[2], dilation=dilation[2])
        self.hidden_conv33 = BasicConv2d(enc_channels*2, channels*2, kernel_size=3, padding=dilation[2], dilation=dilation[2])
        # self.hidden_conv4 = ConvBNReLU(enc_channels, channels, kernel_size=3, stride=1,padding=dilation[3], dilation=dilation[3])
        self.hidden_conv4 = BasicConv2d(enc_channels, channels, kernel_size=3, padding=dilation[3], dilation=dilation[3])
        self.hidden_conv44 = BasicConv2d(enc_channels*2, channels*2, kernel_size=3, padding=dilation[3], dilation=dilation[3])
        # self.conv1 = BasicConv2d(enc_channels*4, channels,1)
        self.conv1 = ConvBNReLU(enc_channels*4, channels,3,stride=1,padding=1)
        self.conv2 = ConvBNReLU(enc_channels*2, channels,1,stride=1,padding=0) # 可能也要改
        self.conv_out = ConvBNReLU(channels, channels, 3,stride=1, padding=1) # 应该要激活
        self.conv_out2 = ConvBNReLU(channels*4, channels, 1,stride=1, padding=0) # 应该要激活
        self.sigmoid = nn.Sigmoid()
    def forward(self,x_t,x_b):#x_t高层特征，x_b低层特征
        expande_xt_up=[] # 顶层上采样
        x_concat = [] # r1,r2,r3,r4
        size = x_b.shape[2:]
        # print(size)
        # 1. 通道扩张为原来的4倍
        x_t_expand = self.scale_expand(x_t)
        # print("x_t_expand.shape",x_t_expand.shape)
        x_b_expand = self.scale_expand(x_b)
        # print("x_b_expand.shape",x_b_expand.shape)
        # 2. 将扩张后的特征分为四组
        expande_xt_split = torch.split(x_t_expand, x_t_expand.size(1) // 4, dim=1)
        expande_xb_split = torch.split(x_b_expand, x_b_expand.size(1) // 4, dim=1)
        for i in range(4):
            expande_xt_up.append(F.interpolate(expande_xt_split[i], size=size, mode='bilinear'))
        # print("expande_xt_up.shape",expande_xt_up[0].shape)
        # print("expande_xb.shape",expande_xb_split[0].shape)
        for i in range(4):
            x_concat.append(torch.cat((expande_xb_split[i],expande_xt_up[i]), dim=1))
        # print("x_concat.shape",x_concat[1].shape)
        # r1 = self.hidden_conv1(x_concat[0])
        # r2 = self.hidden_conv2(x_concat[1])
        # r3 = self.hidden_conv3(x_concat[2])
        # r4 = self.hidden_conv4(x_concat[3])
        # r = torch.cat((r1,r2,r3,r4), dim=1)
        # # print("r.shape",r.shape)
        # r = self.conv1(r) # 3*3卷积输出
        # 相邻特征做乘法
        f1 =self.conv2(x_concat[0])
        f2 =self.conv2(x_concat[1])
        f3 =self.conv2(x_concat[2])
        f4 =self.conv2(x_concat[3])

        r1 = self.hidden_conv1(f1)
        r1_w = self.sigmoid(r1) # r=1的权重
        r2 = self.hidden_conv2(f2*r1_w+f2)
        r2_w = self.sigmoid(r2)
        r3 = self.hidden_conv3(f3*r2_w+f3)
        r3_w = self.sigmoid(r3)
        r4 = self.hidden_conv4(f4*r3_w+f4)
        r = torch.cat((r1,r2,r3,r4), dim=1)
        # print("r.shape",r.shape)
        # r = self.conv1(r) # 3*3卷积输出
        r = self.conv_out2(r) # 1*1卷积输出
        # 仿照resnet
        # r1 = self.hidden_conv1(x_concat[0])
        # r2 = self.hidden_conv22(x_concat[1])
        # r2_cat = self.hidden_conv2(r2)
        # r3 = self.hidden_conv33(x_concat[2]+r2)
        # r3_cat = self.hidden_conv3(r3)
        # r4 = self.hidden_conv44(x_concat[3]+r3)
        # r4_cat = self.hidden_conv4(r4)
        # r = torch.cat((r1,r2_cat,r3_cat,r4_cat), dim=1)
        # # print("r.shape",r.shape)
        # r = self.conv1(r) # 3*3卷积输出
        # print("r_concat.shape",r.shape)
        # concat x_t,x_b
        # 消融实验
        # x_t_up = F.interpolate(x_t, size=size, mode='bilinear')
        # x_concat = torch.cat((x_t_up,x_b), dim=1)
        # x_concat = self.conv2(x_concat)
        # p = x_concat+r
        # p = self.conv_out(p)
        # return x_concat
        return r


class CCM(nn.Module):
    def __init__(self, channel):
        super(CCM, self).__init__()
        self.conv_1 = ConvBNReLU(channel,channel,3,stride=1,padding=1)
        self.conv_3 = ConvBNReLU(2*channel,channel,3,stride=1,padding=1)
        # self.conv_2 = ConvBNReLU(2*channel,channel,3,stride=1,padding=1)
        self.conv_2 = BasicConv2d(2*channel,channel,1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_up, x_down,edge): # x_up是上层特征，x_down是下层特征
        if x_down.size() != edge.size():
            edge = F.interpolate(edge, x_down.size()[2:], mode='bilinear', align_corners=False)
        s_wei = x_up*x_down
        s_wei = self.conv_1(s_wei)
        s_wei = self.avg_pool(s_wei)
        s_wei = self.sigmoid(s_wei)
        x_up_wei = x_up*s_wei
        x_down_wei = x_down*s_wei
        x_concat = torch.cat((x_up_wei, x_down_wei), dim=1)
        x_concat = self.conv_2(x_concat)
        s = x_up+x_down+x_concat
        s = self.conv_1(s)
        # noedge
        s_init = s*edge+s
        s_out = self.conv_1(s_init)
        # return s
        # CCM 消融
        # s_wei = torch.cat((x_up,x_down),dim=1)
        # x_concat = self.conv_3(s_wei)
        # return x_concat
        # return s
        return s_out



if __name__ == "__main__":
    pass
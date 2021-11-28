import math
import torch
import torch.nn as nn
import numpy as np

'''
    模型中各个模块
'''

'''
    CBL模块
'''
class Conv(nn.Module):
    '''
       :param channel_in 输入通道数
       :param channel_out 输出通道数
       :param kernel 卷积核大小
       :param stride 步长
       :param groups 分成几组卷积
       :param activation 是否需要激活函数，默认True
    '''
    def __init__(self, channel_in, channel_out, kernel=1, stride=1, groups=1, activation=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, kernel, stride, kernel // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(channel_out)
        self.act = nn.LeakyReLU(0.1, inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    # 没有bn层的前向传播
    def fuseforward(self, x):
        return self.act(self.conv(x))

'''
    残差模块：Bottleneck结构
'''
class Bottleneck(nn.Module):
    '''
        :param channel_in 输入通道数
        :param channel_out 输出通道数
        :param shortcut 是否需要shortcut连接，默认True
        :param group 分成几组卷积
        :param expansion 扩张，Bottleneck结构中先降维再升维，降多少，用这个参数
    '''
    def __init__(self, channel_in, channel_out, shortcut=True, group=1, expansion=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(channel_out * expansion)    # 降维后通道个数
        self.cv1 = Conv(channel_in, c_, 1, 1)
        self.cv2 = Conv(c_, channel_out, 3, 1, groups=group)
        self.add = shortcut and channel_in == channel_out    # 先降维后升维后的维度和降维前相等，才能做残差连接

    # 前向传播：残差连接
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

'''
    SPP结构：参照yolov3-spp
    先降维，再同时进行三个不同尺度的池化并padding后，和原始结果拼接做升维
'''
class SPP(nn.Module):
    def __init__(self, channel_in, channel_out, kernel=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = channel_in // 2    #
        self.cv1 = Conv(channel_in, c_, 1, 1)    # 先降维
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernel])    # 同时做len(kernel)种maxpooling
        self.cv2 = Conv(c_ * (len(kernel) + 1), channel_out, 1, 1)    # 最后升维

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

# 深度可分离卷积
def DWConv(channel_in, channel_out, kernel=1, stride=1, activation=True):
    return Conv(channel_in, channel_out, kernel, stride, g=math.gcd(channel_in, channel_out), activation=activation)

# Returns x evenly divisble by divisor
def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor

'''
    多分组卷积（深度可分离卷积），每组采用不同的kernel
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
'''
class MixConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel=(1, 3), stride=1, equal_channel=True):
        super(MixConv2d, self).__init__()
        groups = len(kernel)
        if equal_channel:  # 分组卷积，每组相同的channel
            i = torch.linspace(0, groups - 1E-6, channel_out).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [channel_out] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(kernel) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(channel_in, int(c_[g]), kernel[g], stride, kernel[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(channel_out)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))

'''
    Focus模块
'''
class Focus(nn.Module):
    def __init__(self, channel_in, channel_out, kernel=1):
        super(Focus, self).__init__()
        self.conv = Conv(channel_in * 4, channel_out, kernel, 1)

    # x(b, c, w, h) --> y(b, 4d, w/2, h/2)
    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

'''
    Plus-shaped convolution
'''
class ConvPlus(nn.Module):
    def __init__(self, channel_in, channel_out, kernel=3, stride=1, group=1, bias=True):
        super(ConvPlus, self).__init__()
        self.cv1 = nn.Conv2d(channel_in, channel_out, (kernel, 1), stride, (kernel // 2, 0), groups=group, bias=bias)
        self.cv2 = nn.Conv2d(channel_in, channel_out, (1, kernel), stride, (0, kernel // 2), groups=group, bias=bias)

    def forward(self, x):
        return self.cv1(x) + self.cv2(x)

'''
    CSP版Bottleneck
'''
class BottleneckCSP(nn.Module):
    '''
        :param channel_in 输入通道数
        :param channel_out 输出通道数
        :param num Bottleneck结构的个数
        :param shortcut 是否需要shortcut连接，默认True
        :param group 分成几组卷积
        :param expansion 扩张，Bottleneck结构中先降维再升维，降多少，用这个参数
    '''
    def __init__(self, channel_in, channel_out, num=1, shortcut=True, group=1, expansion=0.5):
        super(BottleneckCSP, self).__init__()
        c_ = int(channel_out * expansion)    # Bottleneck结构中降维降到多少
        self.cv1 = Conv(channel_in, c_, 1, 1)    # CBL
        self.cv2 = nn.Conv2d(channel_in, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(channel_out, channel_out, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)    # 准备去cat(conv2, conv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, group, expansion=0.1) for _ in range(num)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

'''
    按指定dim拼接tensor
'''
class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)



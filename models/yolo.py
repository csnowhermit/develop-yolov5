import math
import yaml
import torch
import torch.nn as nn
import numpy as np

import config
import utils.torch_utils as torch_utils
from models.common import Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, ConvPlus, BottleneckCSP, Concat, make_divisible

'''
    Detect模块
'''
class Detect(nn.Module):
    def __init__(self, num_anchors=80, anchors=()):
        super(Detect, self).__init__()
        self.stride = None    # strides computed during build
        self.nc = num_anchors    # number of classes
        self.no = num_anchors + 5    # number of outputs per anchor
        self.nl = len(anchors)    # number of detection layers
        self.na = len(anchors[0]) // 2    # number of anchors
        self.grid = [torch.zeros(1)] * self.nl    # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)    # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))    # shape(nl,1,na,1,1,2)
        self.export = False    # onnx export

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape    # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]    # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]    # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

'''
    总体模型
'''
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 读取配置文件
        with open(config.model_cfg) as f:
            self.model_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # 1.解析配置文件，定义网络
        ch = 3    # 初始通道数为3
        self.model, self.save = parse_model(model_dict=self.model_dict, ch=[ch])

        # 2.构建步长，锚框
        m = self.model[-1]    # 取出Detect()模块
        m.stride = torch.tensor([64 / x.shape[-2] for x in self.forward(torch.zeros(1, ch, 64, 64))])
        m.anchors /= m.stride.view(-1, 1, 1)
        self.stride = m.stride

        # 3.初始化权重weight
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass
            elif t is nn.BatchNorm2d:
                m.eps = 1e-4
                m.momentum = 0.03
            elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True

        # 4.对Detect模块初始化偏置bias
        self._init_biases()

        # 5.打印模型信息
        torch_utils.print_model_info(self)

    '''
        对Detect模块初始化偏置bias
    '''
    def _init_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for f, s in zip(m.ffrom, m.stride):  #  from
            mi = self.model[f % m.i]
            b = mi.bias.view(m.num_anchors, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.num_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x, augment=False):
        if augment:
            img_size = x.shape[-2:]    # height, width
            s = [0.83, 0.67]    # 放缩尺度
            y = []
            # 原图，左右翻转并缩放，直接缩放
            for i, xi in enumerate((x, torch_utils.scale_img(x.flip(3), s[0]), torch_utils.scale_img(x, s[1]), )):
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]    # 缩放
            y[1][..., 0] = img_size[1] - y[1][..., 0]    # 左右翻转
            y[2][..., 4] /= s[1]    # 缩放
            return torch.cat(y, 1), None    # 附带数据增强的推理，训练
        else:
            return self.forward_once(x)    # 单一缩放的推理，训练

    def forward_once(self, x):
        y = []
        for m in self.model:
            if m.f != -1:    # 如果不是来自上一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]    # 来自前面的层（跳跃连接）
            x = m(x)
            y.append(x if m.i in self.save else None)    # 保存输出
        return x

'''
    解析配置文件，构建模型
    :param model_dict 模型文件的读取内容
    :param ch 输入channel列表，一般为[3]
'''
def parse_model(model_dict, ch):
    print('\n%3s%15s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, depth_multiple, width_multiple = model_dict['anchors'], model_dict['nc'], \
                                                           model_dict['depth_multiple'], model_dict['width_multiple']

    # 这里涉及到配置文件里的公式计算，需要和配置文件里写的一模一样
    na = (len(anchors[0]) // 2)  # num of anchors
    no = na * (nc + 5)  # num of outputs = anchors * (classes + 5)。。

    layers, save, channel_out = [], [], ch[-1]  # 层数，保存，输出channel
    for i, (ffrom, num, module, args) in enumerate(model_dict['backbone'] + model_dict['head']):
        # 解析module部分：确保module部分以字符串返回
        module = eval(module) if isinstance(module, str) else module  # eval字符串

        # 解析args部分：确保args部分以字符串返回
        for j, arg in enumerate(args):
            try:
                args[j] = eval(arg) if isinstance(arg, str) else arg  # eval字符串
            except:
                pass

        # 网络深度
        num = max(round(num * depth_multiple), 1) if num > 1 else num

        # 分别处理不同module类型
        if module in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, ConvPlus, BottleneckCSP]:
            channel_in, channel_out = ch[ffrom], args[0]
            channel_out = make_divisible(channel_out * width_multiple, 8) if channel_out != no else channel_out

            args = [channel_in, channel_out, *args[1:]]
            if module is BottleneckCSP:
                args.insert(2, num)
                num = 1
        elif module is nn.BatchNorm2d:
            args = [ch[ffrom]]
        elif module is Concat:
            channel_out = sum([ch[-1 if x == -1 else x + 1] for x in ffrom])
        elif module is Detect:
            ffrom = ffrom or list(
                reversed([(-1 if j == i else j - 1) for j, x in enumerate(ch) if x == no]))
        else:
            channel_out = ch[ffrom]

        # 组合成序列
        module_ = nn.Sequential(*[module(*args) for _ in range(num)]) if num > 1 else module(*args)

        module_.i = i  # 模块序号
        module_.f = ffrom  # 当前模块前面接哪个模块
        module_.type = str(module).replace('__main__.', '')  # 模块类别（名称）
        module_.np = sum([x.numel() for x in module_.parameters()])  # num_params，统计模块的参数量

        print('%3s%15s%3s%10.0f  %-40s%-30s' % (i, ffrom, num, module_.np, module_.type, args))    # 打印当前层结构信息
        save.extend(x % i for x in ([ffrom] if isinstance(ffrom, int) else ffrom) if x != -1)  # 保存列表

        layers.append(module_)  # 追加到模型层列表中
        ch.append(channel_out)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    # with open(config.model_cfg) as f:
    #     model_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    # anchors, num_classes, depth_multiple, width_multiple = model_dict['anchors'], model_dict['nc'], model_dict['depth_multiple'], model_dict['width_multiple']
    # print(len(anchors[0]))
    # print(anchors[0])
    #
    # model = Model()
    # print(model)

    # 推理时直接这样创建并加载模型
    model = torch.load("../checkpoint/yolov5s.pt")['model']
    print("model:", model)
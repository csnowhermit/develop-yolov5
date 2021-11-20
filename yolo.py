import yaml
import torch
import torch.nn as nn

import config

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 读取配置文件
        with open(config.model_cfg) as f:
            self.md = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # 定义网络
        # 1.锚框，类别数量，深度，宽度
        anchors, nc, gd, gw = self.md['anchors'], self.md['nc'], self.md['depth_multiple'], self.md['width_multiple']

        na = (len(anchors[0]) // 2)  # number of anchors
        no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)



if __name__ == '__main__':
    # model = Model().to(config.device)
    with open(config.model_cfg) as f:
        md = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    anchors, nc, gd, gw = md['anchors'], md['nc'], md['depth_multiple'], md['width_multiple']
    print(len(anchors[0]))
    print(anchors[0])

import torch

model_cfg = "./yolov5s.yaml"    # 模型结构配置文件
use_gpu = True
device = torch.device('cuda:0')
weight = "./checkpoint/yolov5s.pt"

# detect 推理过程
source = "./inference/images/"    # 直接写文件目录，如从本地摄像头，直接写"0"
output = "./inference/output/"    # 推理结果保存
half = False    # 不使用半精度推理加速
img_size = 160    # 模型推理时标准输入大小


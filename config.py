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
augment = False    # 是否增强推理
save_txt = False    # 推理结果默认保存
view_img = False    # 显示推理的结果（在图上画出检测框并展示）
save_img = True    # 保存推理的结果（在图上画出检测框并保存）
fourcc = 'mp4v'    # 对于推理的视频，结果保存的格式

# nms过程
conf_thres = 0.4
iou_thres = 0.5
agnostic_nms = False


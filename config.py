import torch

model_cfg = "./yolov5s.yaml"    # 模型结构配置文件
use_gpu = False
device = torch.device('cpu')
weight = "./checkpoint/yolov5s.pt"

# detect 推理过程
source = "0"    # 直接写文件目录，如从本地摄像头，直接写"0"
output = "./inference/output/"    # 推理结果保存
half = False    # 不使用半精度推理加速
img_size = 640    # 模型推理时标准输入大小
augment = False    # 是否增强推理
save_img = True    # 保存推理的结果（在图上画出检测框并保存）

# nms过程
conf_thres = 0.4
iou_thres = 0.5
agnostic_nms = False

# 训练过程
epochs = 300
batch_size = 4
cfg = "./models/balloon/yolov5s.yaml"
train_dataset = "F:/dataset/balloon/images/train/"
val_dataset = "F:/dataset/balloon/images/val/"
nc = 1    # 物体总数
class_names = ['balloon']    # 物体名称

# 超参数
hyp = {'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 5e-4,  # optimizer weight decay
       'giou': 0.05,  # giou loss gain
       'cls': 0.58,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'anchor_t': 4.0,  # anchor-multiple threshold
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.014,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.68,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 0.0,  # image rotation (+/- deg)
       'translate': 0.0,  # image translation (+/- fraction)
       'scale': 0.5,  # image scale (+/- gain)
       'shear': 0.0}  # image shear (+/- deg)
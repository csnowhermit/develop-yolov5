import os
import cv2
import math
import time
import random
import torch
import torchvision
import torch.nn as nn
import numpy as np



'''
    NMS：非极大值抑制
    :param prediction 模型原始结果
    :param conf_thres 物体置信度阈值
    :param iou_thres IOU阈值
    :param fast 快速模式，默认否
'''
def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, fast=False, classes=None, agnostic=False):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """
    num_classes = prediction[0].shape[1] - 5
    xc = prediction[..., 4] > conf_thres    # 先过滤置信度

    # Settings
    min_wh, max_wh = 2, 4096    # (pixels) minimum and maximum box width and height
    max_det = 300    # maximum number of detections per image
    time_limit = 10.0    # seconds to quit after
    redundant = True    # 是否需要冗余的检测，默认是
    fast |= conf_thres > 0.001    # 快速模式
    if fast:
        merge = False
        multi_label = False
    else:
        merge = True    # 合并最好的mAP（每张图片增加0.5ms）
        multi_label = num_classes > 1    # 每个框多个标签

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # 检测矩阵 nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # 按类别过滤
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # 如果没有，则处理下一个图像
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)    # classes
        boxes, scores = x[:, :4] + c, x[:, 4]    # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):    # Merge NMS (boxes merged using weighted mean)
            try:    # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]    # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)    # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]    # require redundancy
            except:    # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break    # time limit exceeded
    return output

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

'''
    检测结果从resize后的坐标还原回原图大小的坐标
'''
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

'''
    在原图上标注推理结果
'''
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1    # 画线/字体的粗细
    color = color or [random.randint(0, 255) for _ in range(3)]    # 如果每传入颜色的话随机指定一个颜色
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))    # 左上右下
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)    #
    if label:
        thickness = max(tl - 1, 1)    # 字体粗细
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3    # 字体涂色区域
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)    # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=thickness, lineType=cv2.LINE_AA)

'''
    格式化时间戳
    :param timestamp time.time()
    :param format 指定格式
    :param ms 是否需要精确到毫秒，默认不需要
'''
def formatTimestamp(timestamp, format="%Y-%m-%d_%H:%M:%S", ms=False):
    time_tuple = time.localtime(timestamp)
    data_head = time.strftime(format, time_tuple)
    if ms is False:
        return data_head
    else:
        data_secs = (timestamp - int(timestamp)) * 1000
        data_ms = "%s.%03d" % (data_head, data_secs)
        return data_ms

'''
    附加类权重：出现次数多的权重大
    :param labels 做成np.array()的标签数据
    :param nc 默认使用coco数据集，共80个类别
'''
def labels_to_class_weights(labels, nc=80):
    if labels[0] is None:
        return torch.Tensor()

    labels = np.concatenate(labels, 0)    # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)    # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)    # 统计每种类别出现的次数

    weights[weights == 0] = 1    # replace empty bins with 1
    weights = 1 / weights    # number of targets per class
    weights /= weights.sum()    # normalize
    return torch.from_numpy(weights)

'''
    计算损失
    :param pred 模型推理的结果
    :param targets 标签
    :param model 模型
'''
def compute_loss(pred, targets, model):
    floatTensor = torch.cuda.FloatTensor if pred[0].is_cuda else torch.Tensor    # 确定tensor的数据类型
    label_cls, label_box, label_obj = floatTensor([0]), floatTensor([0]), floatTensor([0])
    target_cls, target_box, target_obj, anchors = build_targets(pred, targets, model)    # 构建标签集
    loss_reduction = 'mean'

    # 指定loss函数，对pred做sigmoid后再进行BCELoss
    BCE_cls = nn.BCEWithLogitsLoss(pos_weight=floatTensor(model.hyp['cls_pw']), reduction=loss_reduction)    # 相当于带sigmoid的BCELoss
    BCE_obj = nn.BCEWithLogitsLoss(pos_weight=floatTensor(model.hyp['obj_pw']), reduction=loss_reduction)

    # 做标签平滑
    cp, cn = smooth_BCE(eps=0.0)



'''
    构建标签集
    :param pred 预测结果
    :param targets 模型的标签
    :param model moxing 
    :return target_cls 标签的类别列表
    :return target_box 标签的框列表
    :return indices [[image, anchor, grid indices], ...]列表
    :return anchors 标签的anchors框列表
'''
def build_targets(pred, targets, model):
    # Detect()模块
    det = model.module.model[-1] if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel) else model.model[-1]

    na, nt = det.na, targets.shape[0]
    target_cls, target_box, indices, anchors = [], [], [], []
    gain = torch.ones(6, device=targets.device)
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()    # 重叠的偏移量
    at = torch.arange(na).view(na, 1).repeat(1, nt)  # na复制出nt份，at.shape: [na, nt]

    for i in range(det.nl):    # Detect()模块的层数
        anchors = det.anchors[i]
        gain[2:] = torch.tensor(pred[i].shape)[[3, 2, 3, 2]]    # 获取xyxy

        # 匹配目标框和anchors框
        a, t, offsets = [], targets * gain, 0    # offsets为目标框和anchors框的偏移量
        if nt:    # 有目标框才进行如下操作
            r = t[None, :, 4:6] / anchors[:, None]    # 目标框和anchors框的宽高比
            j = torch.max(r, 1./r).max(2)[0] < model.hpy['anchor_t']    # 比较r矩阵，每行的最大值与模型定义中的大小
            a, t = at[j], t.repeat(na, 1, 1)[j]

            # 计算重叠部分
            gxy = t[:, 2:4]    # [len(a), 2]
            z = torch.zeros_like(gxy)
            g = 0.5  # offset
            j, k = ((gxy % 1. < g) & (gxy > 1.)).cpu().numpy().T
            l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).cpu().numpy().T
            a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
            offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

        b, c = t[:, :2].cpu().numpy().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.cpu().numpy().T  # grid xy indices

        indices.append((b, a, gj, gi))    # image, anchor, grid indices
        target_box.append(torch.cat((gxy.float() - gij.float(), gwh.float()), 1))
        anchors.append(anchors[a])
        target_cls.append(c)
    return target_cls, target_box, indices, anchors

'''
    做标签平滑
'''
def smooth_BCE(eps=0.0):
    return 1.0 - 0.5 * eps, 0.5 * eps

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

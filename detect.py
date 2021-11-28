import os
import cv2
import time
import shutil
import colorsys
import torch
import numpy as np
from sys import platform
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw

import config
from utils.dataset import LoadStream, LoadImages
from utils.util import non_max_suppression, xyxy2xywh, scale_coords

'''
    推理过程
'''

if __name__ == '__main__':
    with torch.no_grad():
        model = torch.load(config.weight, map_location=config.device)['model']
        model.to(config.device).eval()

        if os.path.exists(config.output):
            shutil.rmtree(config.output)
        os.makedirs(config.output)

        # 只有cuda环境下才能使用半精度
        if config.half is True and config.use_gpu is True:
            model.half()

        # 将输入数据集做成DataLoader
        webcam = config.source == '0' or config.source.startswith('rtsp') or config.source.startswith('http') or config.source.endswith('.txt')
        if webcam:
            if config.use_gpu:
                torch.backends.cudnn.benchmark = True
            dataset = LoadStream(config.source)
        else:
            dataset = LoadImages(config.source)

        # 设置颜色
        class_names = model.names if hasattr(model, 'names') else model.modules.names    # 总类别列表

        hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # 开始推理
        t0 = time.time()
        # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        # _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
        # img为resize后channel first的，im0s是原始图像
        for path, img, im0s, cap in dataset:
            img = torch.from_numpy(img).to(config.device)
            img = img.half() if config.half else img.float()    # 半精度计算
            img /= 255.0    # 归一化
            if img.ndimension() == 3:
                img = img.unsqueeze(0)    # 新增一个batch维度，做成BCHW的

            # 推理
            t1 = time.time()
            pred = model(img, augment=config.augment)[0]
            t2 = time.time()

            # nms
            if config.half:
                pred = pred.float()

            # Apply NMS。pred: list(tensor)
            pred = non_max_suppression(pred, conf_thres=config.conf_thres, iou_thres=config.iou_thres,
                                       fast=True, classes=None, agnostic=config.agnostic_nms)

            # 根据每张图片大小设置字体和画框的粗细
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * im0s.shape[0] + 0.5).astype('int32'))
            thickness = (im0s.shape[0] + im0s.shape[1]) // 300

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                # det = det.detach().cpu().numpy()    # 将pred中每个元素都转成np.ndarray，方便后处理
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()    # im0:np.ndarray
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(config.output) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                im0_pil = Image.fromarray(im0)    # 做成PIL.Image，供ImageDraw用
                if det is not None and len(det):
                    # 放缩图像大小由img到原始shape
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        # print("c:", type(c), c)
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, class_names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        if config.save_txt:    # 检测结果保存到文件
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                                file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if config.save_img or config.view_img:
                            label = '%s %.2f' % (class_names[int(cls)], conf)

                            draw = ImageDraw.Draw(im0_pil)
                            label_size = draw.textsize(label, font)

                            left, top, right, bottom = xyxy    # 这里是左上右下。在gpu设备上
                            top = max(0, np.floor(top.item() + 0.5).astype('int32'))    # 使用.item()将数据转移到cpu上才能继续计算，.cpu()无效
                            left = max(0, np.floor(left.item() + 0.5).astype('int32'))
                            bottom = min(im0.shape[1], np.floor(bottom.item() + 0.5).astype('int32'))
                            right = min(im0.shape[0], np.floor(right.item() + 0.5).astype('int32'))
                            # print("\t", label, (left, top), (right, bottom))

                            if top - label_size[1] >= 0:
                                text_origin = np.array([left, top - label_size[1]])
                            else:
                                text_origin = np.array([left, top + 1])

                            # My kingdom for a good redistributable image drawing library.
                            for i in range(thickness):
                                draw.rectangle(
                                    [left + i, top + i, right - i, bottom - i],
                                    outline=colors[int(c.item())])
                            draw.rectangle(
                                [tuple(text_origin), tuple(text_origin + label_size)],
                                fill=colors[int(c.item())])
                            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                            del draw
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # 标注完之后再转回np.ndarray
                im0 = np.array(im0_pil)

                # Stream results
                if config.view_img:
                    cv2.imshow(p, np.ndarray(im0))
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # 保存推理结果
                if config.save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(im0, "YOLO v5 | by Xujing | Tesla V100 32G", (40, 40), font, 0.7,
                                    (0, 255, 0), 2)
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fps = cap.get(cv2.CAP_PROP_FPS)
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(config.fourcc), fps, (w, h))
                        vid_writer.write(im0)

        if config.save_txt or config.save_img:
            print('Results saved to %s' % os.getcwd() + os.sep + config.output)
            if platform == 'darwin':  # MacOS
                os.system('open ' + save_path)

        print('Done. (%.3fs)' % (time.time() - t0))




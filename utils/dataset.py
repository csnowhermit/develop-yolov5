import os
import cv2
import time
import glob
from pathlib import Path
import numpy as np
from threading import Thread

# 图片和视频支持如下格式：
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
vid_formats = ['.mov', '.avi', '.mp4']


'''
    视频流：目前只支持单一摄像头，一次只处理一张图片
'''
class LoadStream:
    def __init__(self, sources='streams.txt', img_size=416):
        self.mode = 'stream'
        self.img_size = img_size

        # 解析视频流数据源
        if os.path.isfile(sources) is True:
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.readlines() if len(x.strip())]
        else:
            sources = [sources]

        # 多个数据源同时取数
        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, source in enumerate(sources):
            # 逐个开启线程取数
            print("%g/%g: %s..." % (i+1, n, source), end=' ')
            cap = cv2.VideoCapture(0 if source == '0' else source)    # 读取本地摄像头用的是数字0，而不是字符串'0'
            assert cap.isOpened(), 'Failed to open %s' % source

            # 获取宽、高、FPS
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100

            _, self.imgs[i] = cap.read()
            thread = Thread(target=self.update, args=[i, cap], daemon=True)    # 新开个线程，放后台读取数据
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print("")

        # 检查各图像的shape
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        cnt = 0
        while cap.isOpened():
            cnt += 1
            cap.grab()    # 抓取图像
            if cnt == 4:    # 每四帧取一帧
                _, self.imgs[index] = cap.retrieve()
                cnt = 0
            time.sleep(0.01)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):    # 按q键退出
            cv2.destroyAllWindows()
            raise StopIteration

        # letterbox方式做resize
        img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0][0]    # 一帧一帧做

        img = img[:, :, ::-1].transpose(2, 0, 1)    # BHWC-->BCHW, BGR-->RGB
        img = np.ascontiguousarray(img)    # 将内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        return self.sources, img, img0[0], None

    def __len__(self):
        return 0


'''
    图像集
'''
class LoadImages:
    def __init__(self, path, img_size=416):
        path = str(Path(path))
        fileList = []
        if os.path.isdir(path):
            fileList = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            fileList = [path]

        images = [x for x in fileList if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in fileList if os.path.splitext(x)[-1].lower() in vid_formats]
        num_images, num_videos = len(images), len(videos)

        self.img_size = img_size
        self.fileList = images + videos
        self.num_files = num_images + num_videos
        self.video_flag = [False] * num_images + [True] * num_videos    # 标注当前文件是否是视频

        self.mode = 'images'

        if any(videos):    # 单独处理视频文件
            self.new_video(videos[0])
        else:
            self.cap = None

        assert self.num_files > 0, 'No images or videos found in ' + path

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.num_files:
            raise StopIteration
        path = self.fileList[self.count]
        if self.video_flag[self.count]:
            self.mode = 'video'
            ret, img0 = self.cap.read()
            if not ret:    # 如果读取失败，就开始读下一个文件
                self.count += 1
                self.cap.release()
                if self.count == self.num_files:    # 最后一个video
                    raise StopIteration
                else:
                    path = self.fileList[self.count]
                    self.new_video(path)
                    ret, img0 = self.cap.read()
            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.num_files, self.frame, self.num_frames, path), end='')
        else:
            self.count += 1
            img0 = cv2.imread(path)    # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.num_files, path), end='')

        # letterbox方式resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR-->RGB
        img = np.ascontiguousarray(img)  # 将内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.num_files    # 文件个数





'''
    采用letterbox方式做resize，resize的同时也考虑宽高比
'''
def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]    # 原始shape：hw
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    scale_ratio = min(new_shape[0]/shape[0], new_shape[1]/shape[1])    # 确定放缩比率
    if not scaleup:    # 仅仅向下放缩，不向上放缩
        scale_ratio = min(scale_ratio, 1.0)

    # 计算填充方式
    ratio = scale_ratio, scale_ratio    # h，w的各种方式
    new_unpad = int(round(shape[0] * scale_ratio)), int(round(shape[1] * scale_ratio))    # 计算新的shape, hw
    dh, dw = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]    # 要填充的大小，hw

    if auto:    # 自动，框住最小区域矩形
        dh, dw = np.mod(dh, 64), np.mod(dw, 64)    # np.mod()模运算
    elif scaleFill:
        dh, dw = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0]/shape[0], new_shape[1]/shape[1]    # 拉伸的话重新计算放缩比率，hw

    # 上下和左右均等填充
    dh /= 2
    dw /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, (new_unpad[1], new_unpad[0]), interpolation=cv2.INTER_LINEAR)    # resize时dsize为(w, h)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)    # 上下左右用相同颜色填充
    return img, ratio, (dh, dw)

if __name__ == '__main__':
    img = np.ndarray([100, 150, 3])
    img2, ratio, (dh, dw) = letterbox(img, 200)
    print(img2.shape)
    print(ratio)
    print(dh, dw)


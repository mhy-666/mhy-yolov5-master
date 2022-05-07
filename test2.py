from flask import Flask, render_template, Response
import cv2
import time
import logging
import numpy as np
import onnxruntime

import argparse
import os
import sys
from pathlib import Path

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


# load onnx model
def load_model(onnx_model):
    sess = onnxruntime.InferenceSession(onnx_model)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    logging.info("输入的name:{}, 输出的name:{}".format(in_name, out_name))

    return sess, in_name, out_name


# process frame
def frame_process(frame, input_shape=(416, 416)):
    img = cv2.resize(frame, input_shape)
    image = img[:, :, ::-1].transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :] / 255
    image = np.array(image, dtype=np.float32)
    return image

def nms(pred, conf_thres, iou_thres):
    # 置信度抑制，小于置信度阈值则删除
    conf = pred[..., 4] > conf_thres
    box = pred[conf == True]
    # 类别获取
    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))
    # 获取类别
    total_cls = list(set(cls))  #删除重复项，获取出现的类别标签列表,example=[0, 17]
    output_box = []   #最终输出的预测框
    # 不同分类候选框置信度
    for i in range(len(total_cls)):
        clss = total_cls[i]   #当前类别标签
        # 从所有候选框中取出当前类别对应的所有候选框
        cls_box = []
        for j in range(len(cls)):
            if cls[j] == clss:
                box[j][5] = clss
                cls_box.append(box[j][:6])
        cls_box = np.array(cls_box)
        box_conf = cls_box[..., 4]   #取出候选框置信度
        box_conf_sort = np.argsort(box_conf)   #获取排序后索引
        max_conf_box = cls_box[box_conf_sort[len(box_conf) - 1]]
        output_box.append(max_conf_box)   #将置信度最高的候选框输出为第一个预测框
        cls_box = np.delete(cls_box, 0, 0)  #删除置信度最高的候选框
        while len(cls_box) > 0:
            max_conf_box = output_box[len(output_box) - 1]     #将输出预测框列表最后一个作为当前最大置信度候选框
            del_index = []
            for j in range(len(cls_box)):
                current_box = cls_box[j]      #当前预测框
                interArea = getInter(max_conf_box, current_box)    #当前预测框与最大预测框交集
                iou = getIou(max_conf_box, current_box, interArea)  # 计算交并比
                if iou > iou_thres:
                    del_index.append(j)   #根据交并比确定需要移出的索引
            cls_box = np.delete(cls_box, del_index, 0)   #删除此轮需要移出的候选框
            if len(cls_box) > 0:
                output_box.append(cls_box[0])
                cls_box = np.delete(cls_box, 0, 0)
    return output_box

app = Flask(__name__)
height, width = 640, 640
img0 = cv2.imread('bus.jpg')
print(img0.shape)
img = cv2.resize(img0, (height, width))  # 尺寸变换
img = img / 255.
img = img[:, :, ::-1].transpose((2, 0, 1))  # HWC转CHW
data = np.expand_dims(img, axis=0)  # 扩展维度至[1,3,640,640]
sess,input_name,output_name = load_model('yolov5s.onnx')
pred_onx = sess.run([output_name], {input_name: data.astype(np.float32)})[0]
pred = np.squeeze(pred_onx)
print(pred)
print(pred.shape)
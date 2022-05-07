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
import test4 as onnx_interface


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
    in_name = [input.name for input in sess.get_inputs()][0]
    out_name = [output.name for output in sess.get_outputs()]
    logging.info("输入的name:{}, 输出的name:{}".format(in_name, out_name))

    return sess, in_name, out_name


# process frame
def frame_process(frame, input_shape=(416, 416)):
    img = cv2.resize(frame, input_shape)
    image = img[:, :, ::-1].transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :] / 255
    image = np.array(image, dtype=np.float32)
    return image


app = Flask(__name__)

# 视屏预处理
def stream_inference(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=0,  # file/dir/URL/glob, 0 for webcam"http://admin:admin@100.73.95.209:8081"
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu tensor_rt
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = onnx_interface.YOLOV5_ONNX(onnx_path="yolov5s.onnx")


    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    height, width = 640, 640
    # Run inference

    for path, im, im0s, vid_cap, s in dataset:
        start = time.time()
        # print(path)
        # print(im.shape)
        # print(im0s)
        # print(vid_cap)
        # print(s)
        print(im.shape)
        print(im.dtype)



        # 超参数设置
        img_size = (640, 640)  # 图片缩放大小
        src_img = im[0, :, :, :]
        # print(src_img.shape)
        start = time.time()
        src_size = src_img.shape[:2]
        src_img = src_img[::-1, :, :].transpose((1, 2, 0))
        # print(src_img.shape)
        img = cv2.resize(src_img, img_size)


        print("start: ",time.time())
        image=model.infer(img_path=img)
        print("end: ",time.time())

        if type(image) is np.ndarray:
            img=image
        else:
            img=img.astype(np.uint8)
        # print(img)
        # img1 = np.clip(img, 0, 255).astype(np.uint8)
        # img2=img1.copy()
        # cv2.imwrite("res2.jpg",img2)

        ret, buffer = cv2.imencode('.jpg', img)


        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # print(start)



@app.route('/video_start')
def video_start():

    # 通过将一帧帧的图像返回，就达到了看视频的目的。multipart/x-mixed-replace是单次的http请求-响应模式，如果网络中断，会导致视频流异常终止，必须重新连接才能恢复
    return Response(stream_inference(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


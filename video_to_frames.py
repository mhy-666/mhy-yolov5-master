import cv2
import os

def video_to_frames(video_path, outPutDirName):
    times = 0
    # 提取视频的频率，每30帧提取一个
    frame_frequency = 120
    # 如果文件目录不存在则创建目录
    if not os.path.exists(outPutDirName):
        os.makedirs(outPutDirName)
    # 读取视频帧
    camera = cv2.VideoCapture(video_path)
    while True:
        times = times + 1
        res, image = camera.read()
        if not res:
            print('not res , not image')
            break
        # 按照设置间隔存储视频帧
        if times % frame_frequency == 0:
            cv2.imwrite(outPutDirName + '\\' + str(times) + '.jpg', image)
    print('图片提取结束')
    # 释放摄像头设备
    camera.release()

video_to_frames("supermarket.mp4", "./frames")
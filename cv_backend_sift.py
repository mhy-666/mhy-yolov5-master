import cv2
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import filters
from scipy import spatial
import time
from imutils import paths
import os
start=time.time()

def adjust_colour(images_temp, shift, bestIndex, gamma=2.2):  # set gamma to 2.2 by paper
    alpha = np.ones((3, len(images_temp)))

    # compute light averages in the overlap area by linearizing the gamma-corrected RGB values
    for rightBorder in range(bestIndex + 1, len(images_temp)):
        for i in range(bestIndex + 1, rightBorder + 1):
            I = images_temp[i]
            J = images_temp[i - 1]
            overlap = I.shape[1] - shift[i - 1]
            for channel in range(3):
                alpha[channel, i] = np.sum(np.power(J[:, -overlap - 1:, channel], gamma)) / np.sum(
                    np.power(I[:, 0:overlap + 1, channel], gamma))  # derivative

        G = np.sum(alpha, 1) / np.sum(np.square(alpha), 1)

        for i in range(bestIndex + 1, rightBorder + 1):
            for channel in range(3):
                images_temp[i][:, :, channel] = np.power(G[channel] * alpha[channel, i], 1.0 / gamma) * images_temp[i][
                                                                                                        :, :,
                                                                                                        channel]  # perform using correction coefficients and the global adjustment

    for leftBorder in range(bestIndex - 1, -1, -1):
        for i in range(bestIndex - 1, leftBorder - 1, -1):
            I = images_temp[i]
            J = images_temp[i + 1]
            overlap = I.shape[1] - shift[i - 1]
            for channel in range(3):
                alpha[channel, i] = np.sum(np.power(J[:, 0:overlap + 1, channel], gamma)) / np.sum(
                    np.power(I[:, -overlap - 1:, channel], gamma))

        G = np.sum(alpha, 1) / np.sum(np.square(alpha), 1)

        for i in range(bestIndex - 1, leftBorder - 1, -1):
            for channel in range(3):
                images_temp[i][:, :, channel] = np.power(G[channel] * alpha[channel, i], 1.0 / gamma) * images_temp[i][
                                                                                                        :, :, channel]
    return images_temp

def get_sift_image(
        path1,
        path2
):
    # img1=cv2.imread(path1)
    # img2=cv2.imread(path2)
    img1=path1
    img2=path2
    newimg1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    newimg2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # extract descriptor
    sift = cv2.SIFT_create()
    kps1, des1 = sift.detectAndCompute(newimg1, None)
    kps2, des2 = sift.detectAndCompute(newimg2, None)

    # knn feature matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    choic = []
    for m in matches:
        if (m[0].distance < 0.5 * m[1].distance):
            choic.append(m)
    matches = np.asarray(choic)

    # ransac Linear fit
    if (len(matches[:, 0]) >= 4):
        src = np.float32([kps1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32([kps2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    else:
        raise AssertionError('no enough key points')

    dst = cv2.warpPerspective(img1, H, ((img1.shape[1] + img2.shape[1]), img2.shape[0]))  # wraped image
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2  # stitched image
    end = time.time()
    dur = end - start
    print(dur)
    cv2.imwrite('output.jpg', dst)
    # plt.imshow(dst)
    # plt.show()
    return dst

def video_to_frames(video_path, outPutDirName,frame_frequency= 150):
    times = 0
    # 提取视频的频率，每30帧提取一个
    # 如果文件目录不存在则创建目录
    if not os.path.exists(outPutDirName):
        os.makedirs(outPutDirName)
    # 读取视频帧
    camera = cv2.VideoCapture(video_path)
    count=-1
    while True:
        times = times + 1
        res, image = camera.read()
        if not res:
            print('not res , not image')
            break
        # 按照设置间隔存储视频帧
        if times % frame_frequency == 0:
            count=count+1
            outPutDirName=str(outPutDirName)
            cv2.imwrite(outPutDirName +'/'+ str(count) + '.jpg', image)
    print('图片提取结束')
    # 释放摄像头设备
    camera.release()

if __name__=="__main__":
    imgs_dir = r'M:\pycharmProjects\mhy-yolov5-master\ouputImage-Panorama-Stitching\sift_test'
    video_to_frames("qingmingshanghetunew.mp4",imgs_dir,150)
    imagePaths = sorted(list(paths.list_images(imgs_dir)))
    print(imagePaths)
    images = []

    # loop over the image paths, load each one, and add them to our
    # images to stich list
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        images.append(image)


    count=0
    print(images[count])
    temp_image=images[count]
    while True:
        if count==len(images)-2:
            break
        else:
            temp_image=get_sift_image((temp_image),(images[count+1]))

    # for img in images:
    #     if



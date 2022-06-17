import glob
import numpy as np
from PIL import Image
import imageio
from  matplotlib import pyplot as plt
import cv2
import os
from pathlib import Path
import re



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]




def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path



def video_to_frames(video_path, outPutDirName):
    times = 0
    # 提取视频的频率，每30帧提取一个
    frame_frequency = 40
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
            outPutDirName=str(outPutDirName)
            cv2.imwrite(outPutDirName +'/'+ str(times) + '.jpg', image)
    print('图片提取结束')
    # 释放摄像头设备
    camera.release()




def get_proper_image_count(imgs):
    best_index = 0
    best_value = 255
    for index, img in enumerate(imgs):
        current_mean = np.array([np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])])    # average for three channels
        diff = np.max(current_mean) - np.min(current_mean)  # difference of three channels
        if diff < best_value:       # choose the best(lowest)
            best_index = index
            best_value = diff
    return best_index



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



# imgs_corrected = imgs

def get_surfError(panorama, curr_img, overlap, channel):
    left = panorama[:, -overlap-1:, channel]
    right = curr_img[:, 0:overlap+1, channel]
    return np.square(left - right)

def get_minMSE(e):
    E = np.zeros(e.shape)   # cumulative minimum squared difference
    E[0,:] = e[0,:]
    # dynamic programming
    for h in range(1, e.shape[0]):
        for w in range(0, e.shape[1]):
            if w == 0:
                cost = min(E[h-1, w], E[h-1, w+1])
            elif w == e.shape[1]-1:
                cost = min(E[h-1, w-1], E[h-1, w])
            else:
                cost = min(E[h-1, w-1], E[h-1, w], E[h-1, w+1])
            E[h,w] = e[h,w] + cost
    return E

def get_minMSEPath(E, e):
    h = e.shape[0]
    path = np.zeros((h, 1))
    idx = np.argmin(E[h-1, :])
    path[h-1] = idx
    for h in range(e.shape[0]-2,-1,-1):     # tracking back the paths with a minimal cost from bottom to top
        w = int(path[h+1][0])
        if w > 0 and E[h, w-1] == E[h+1, w]-e[h+1, w]:
            path[h] = w-1
        elif w < e.shape[1] - 1 and E[h, w+1] == E[h+1, w]-e[h+1, w]:
            path[h] = w+1
        else:
            path[h] = w

    path[path==0] = 1
    return path


def get_imageStitched(panorama, curr_img, path, overlap):
    n = 1
    bound_threshold = 15  # poisson blending threshold(hyper parameter)

    resultImg = np.zeros((0, panorama.shape[1] + curr_img.shape[1] - overlap, 3)).astype('float64')
    for h in range(0, panorama.shape[0]):
        A = np.expand_dims(panorama[h, 0:-(overlap - int(path[h][0]) + 1), :], axis=0)
        B = np.expand_dims(curr_img[h, int(path[h][0]) - 1:, :], axis=0)

        # deriveration
        ZA = np.concatenate((np.expand_dims(panorama[h, :, :], axis=0), np.zeros((A.shape[0],
                                                                                  panorama.shape[1] + curr_img.shape[
                                                                                      1] - overlap -
                                                                                  np.expand_dims(panorama[h, :, :],
                                                                                                 axis=0).shape[1], 3))),
                            axis=1)
        ZB = np.concatenate((np.expand_dims(panorama[h, 0:panorama.shape[1] + curr_img.shape[1] - overlap -
                                                          np.expand_dims(curr_img[h, :, :], axis=0).shape[1], :],
                                            axis=0), np.expand_dims(curr_img[h, :, :], axis=0)), axis=1)

        filter_A = np.ones((1, A.shape[1] - bound_threshold))
        grad = np.expand_dims(np.linspace(1, 0, 2 * bound_threshold + 1, endpoint=True), axis=0)
        filter_B = np.zeros((1, B.shape[1] - bound_threshold))
        blender = np.concatenate((filter_A, grad, filter_B), axis=1)
        Z = (blender[:, 0:ZA.shape[1]].T * ZA.T).T + ((1 - blender[:, 0:ZB.shape[1]]).T * ZB.T).T

        resultImg = np.concatenate((resultImg, Z))
    return resultImg





def transformVideo(
        path="test2.mp4",
        outputPath="./ouputImage-Panorama-Stitching",
        project=ROOT / 'ouputImage-Panorama-Stitching',
        name='video_result'):
    save_dir = increment_path(Path(project) / name, exist_ok=False)
    video_to_frames(path, save_dir)
    imgs_dir = str(save_dir)
    print(imgs_dir)
    files = glob.glob(imgs_dir + '\*.*g')  # list of matching file paths
    files = sorted(files)
    print(files)
    imgs = [np.array(Image.open(files[i])) for i in range(len(files))]

    beta = 0
    overlapArea = [imgs[0].shape[1] // (3 + beta)] * (len(imgs) - 1)  # overlap range(hyper parameter)
    proper_image_count = get_proper_image_count(imgs)
    imgs_corrected = adjust_colour(imgs, overlapArea, proper_image_count)
    panorama = imgs_corrected[0]
    for i in range(1, len(imgs_corrected)):
        curr_img = imgs_corrected[i]
        channel = np.argmax([np.var(curr_img[:, :, 0]), np.var(curr_img[:, :, 1]),
                             np.var(curr_img[:, :, 2])])  # get the channel with the largest mean variance

        overlap = curr_img.shape[1] - overlapArea[i - 1]

        error_surface = get_surfError(panorama, curr_img, overlap, channel)

        # imageio.imwrite(imgs_dir+'error_surface.png', error_surface)

        E = get_minMSE(error_surface)

        # imageio.imwrite(imgs_dir+'E.png', E)

        path = get_minMSEPath(E, error_surface)

        # imageio.imwrite(imgs_dir+'path.png', path)

        panorama = get_imageStitched(panorama, curr_img, path, overlap)

    result = np.array(255 * panorama / np.max(panorama)).astype('uint8')
    plt.figure()
    plt.imshow(result)
    imageio.imwrite(imgs_dir + '/output2.png', np.array(255 * panorama / np.max(panorama)).astype('uint8'))
    url = imgs_dir + '/output2.png'
    return url

def transformImage(
        path=r'M:\pycharmProjects\mhy-yolov5-master\ouputImage-Panorama-Stitching\test',
        outputPath="./ouputImage-Panorama-Stitching",
        project=ROOT / 'ouputImage-Panorama-Stitching',
        name='image_result'):
    save_dir = increment_path(Path(project) / name, exist_ok=False)
    
    imgs_dir = path
    print(imgs_dir)
    files = glob.glob(imgs_dir + '\*.*g')  # list of matching file paths
    files = sorted(files)
    print(files)
    imgs = [np.array(Image.open(files[i])) for i in range(len(files))]

    beta = 0
    overlapArea = [imgs[0].shape[1] // (3 + beta)] * (len(imgs) - 1)  # overlap range(hyper parameter)
    proper_image_count = get_proper_image_count(imgs)
    adjustImage = adjust_colour(imgs, overlapArea, proper_image_count)
    panorama = adjustImage[0]
    for i in range(1, len(adjustImage)):
        curr_img = adjustImage[i]
        channel = np.argmax([np.var(curr_img[:, :, 0]), np.var(curr_img[:, :, 1]),
                             np.var(curr_img[:, :, 2])])  # get the channel with the largest mean variance

        overlap = curr_img.shape[1] - overlapArea[i - 1]

        error_surface = get_surfError(panorama, curr_img, overlap, channel)

        # imageio.imwrite(imgs_dir+'error_surface.png', error_surface)

        E = get_minMSE(error_surface)

        # imageio.imwrite(imgs_dir+'E.png', E)

        path = get_minMSEPath(E, error_surface)

        # imageio.imwrite(imgs_dir+'path.png', path)

        panorama = get_imageStitched(panorama, curr_img, path, overlap)

    result = np.array(255 * panorama / np.max(panorama)).astype('uint8')
    plt.figure()
    plt.imshow(result)
    imageio.imwrite(imgs_dir + '/outputResult.png', np.array(255 * panorama / np.max(panorama)).astype('uint8'))
    url = imgs_dir + '/outputResult.png'
    return url

def getImageAndSitch(
        ifFirst=False,
        flag=False,
        frontImg='',
        project=ROOT / 'ouputImage-Panorama-Stitching',
        name='image_result'
):
    if ifFirst==True:
        save_dir = increment_path(Path(project) / name, exist_ok=False)
        
    else:
        if flag == False:
            flag=flag
    # 把传来的图片放入save_dir
        else:
            transformImage()
        
        

if __name__=="__main__":
    transformImage()

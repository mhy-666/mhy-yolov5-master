import glob
from PIL import Image
import imageio
from  matplotlib import pyplot as plt
import cv2
import os
from pathlib import Path
import re
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]



#Automatically add folder
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


# Extract video frames
def video_to_frames(video_path, outPutDirName,frame_frequency= 150):
    times = 0
    # Extract video frequency, one every 150 frames
    # Create directory if file directory does not exist
    if not os.path.exists(outPutDirName):
        os.makedirs(outPutDirName)
    # Read video frame
    camera = cv2.VideoCapture(video_path)
    while True:
        times = times + 1
        res, image = camera.read()
        if not res:
            print('not res , not image')
            break
        # Store video frames at set intervals
        if times % frame_frequency == 0:
            outPutDirName=str(outPutDirName)
            cv2.imwrite(outPutDirName +'/'+ str(times) + '.jpg', image)
    print('over')
    # Release camera device
    camera.release()


def transformVideo(
        path="qingmingshanghetunew.mp4",
        project=ROOT / 'ouputImage-Panorama-Stitching',
        name='video_result',
        if_Remove_Black=1,
        frame_frequency=150
        ):
    url = []
    save_dir = increment_path(Path(project) / name, exist_ok=False)
    video_to_frames(path, save_dir,frame_frequency)
    imgs_dir = str(save_dir)
    print(imgs_dir)
    # files = glob.glob(imgs_dir + '\*.*g')  # list of matching file paths
    # files = sorted(files)
    # print(files)
    # imgs = [np.array(Image.open(files[i])) for i in range(len(files))]
    # grab the paths to the input images and initialize our images list
    print("[INFO] loading images...")
    image_paths = sorted(list(paths.list_images(imgs_dir)))

    images = []

    # append images to list
    for image in image_paths:
        image = cv2.imread(image)
        images.append(image)

    print("[INFO] stitching images...")
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitchedImage) = stitcher.stitch(images)

    print(status)
    # status 0 means success
    if status == 0:
        #whether remove black border
        if if_Remove_Black > 0:

            stitchedImage = cv2.copyMakeBorder(stitchedImage, 2, 2, 2, 2,
                                          cv2.BORDER_CONSTANT, (0, 0, 0))

            # Convert to grayscale image and set threshold
            gray = cv2.cvtColor(stitchedImage, cv2.COLOR_BGR2GRAY)
            threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

            # Find all characteristic points under the set threshold
            characteristic_point = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            characteristic_point = imutils.grab_contours(characteristic_point)
            c = max(characteristic_point, key=cv2.contourArea)

            # Stitch rectangular bounding box of image area
            mask = np.zeros(threshold.shape, dtype="uint8")
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

            rect_area = mask.copy()
            temp = mask.copy()

            # subtracted image
            while cv2.countNonZero(temp) > 0:
                # remove the black region
                rect_area = cv2.erode(rect_area, None)
                temp = cv2.subtract(rect_area, threshold)

            # get the bounding box
            characteristic_point = cv2.findContours(rect_area.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            characteristic_point = imutils.grab_contours(characteristic_point)
            c = max(characteristic_point, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)

            # get the final image
            stitchedImage = stitchedImage[y:y + h, x:x + w]

        # Save picture
        # url = save_dir + '/outputResult.png'
        url='./result/outputResult.png'
        cv2.imwrite(url, stitchedImage)


        cv2.imshow("stitchedImage", stitchedImage)
        cv2.waitKey(0)

    # if there doesn't exist enough keypoints
    else:
        print("[INFO] image stitching failed ({})".format(status))
    # print(url)
    return url,status

def transformImage(
        path=r'M:\pycharmProjects\mhy-yolov5-master\ouputImage-Panorama-Stitching\test2',
        project=ROOT / 'ouputImage-Panorama-Stitching',
        name='image_result',
        if_Remove_Black=1):
    url=[]
    save_dir = increment_path(Path(project) / name, exist_ok=False)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    imgs_dir = path
    print("[INFO] loading images...")
    image_paths = sorted(list(paths.list_images(imgs_dir)))

    images = []

    # append images to list
    for image in image_paths:
        image = cv2.imread(image)
        images.append(image)


    print("[INFO] stitching images...")
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitchedImage) = stitcher.stitch(images)

    # status 0 means success
    if status == 0:
        #whether remove black border
        if if_Remove_Black > 0:

            stitchedImage = cv2.copyMakeBorder(stitchedImage, 2, 2, 2, 2,
                                               cv2.BORDER_CONSTANT, (0, 0, 0))

            # Convert to grayscale image and set threshold
            gray = cv2.cvtColor(stitchedImage, cv2.COLOR_BGR2GRAY)
            threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

            # Find all characteristic points under the set threshold
            characteristic_point = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            characteristic_point = imutils.grab_contours(characteristic_point)
            c = max(characteristic_point, key=cv2.contourArea)

            # Stitch rectangular bounding box of image area
            mask = np.zeros(threshold.shape, dtype="uint8")
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

            rect_area = mask.copy()
            temp = mask.copy()

            # subtracted image
            while cv2.countNonZero(temp) > 0:
                # remove the black region
                rect_area = cv2.erode(rect_area, None)
                temp = cv2.subtract(rect_area, threshold)

            # get the bounding box
            characteristic_point = cv2.findContours(rect_area.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            characteristic_point = imutils.grab_contours(characteristic_point)
            c = max(characteristic_point, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)

            # get the final image
            stitchedImage = stitchedImage[y:y + h, x:x + w]

        # Save picture
        save_dir = str(save_dir)
        print()
        # url = save_dir + '/outputResult.png'
        url='./result/outputResult.png'
        print(url)
        cv2.imwrite(url, stitchedImage)

        cv2.imshow("stitchedImage", stitchedImage)
        cv2.waitKey(0)

    # if there doesn't exist enough keypoints
    else:
        print("[INFO] image stitching failed ({})".format(status))
    return url,status
# def getImageAndSitch(
#         ifFirst=False,
#         flag=False,
#         frontImg='',
#         project=ROOT / 'ouputImage-Panorama-Stitching',
#         name='image_result'
# ):
#     if ifFirst==True:
#         save_dir = increment_path(Path(project) / name, exist_ok=False)
#
#     else:
#         if flag == False:
#             flag=flag

#         else:
#             transformImage()
        
        

if __name__=="__main__":
    transformImage()

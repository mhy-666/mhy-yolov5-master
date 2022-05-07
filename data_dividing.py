import os
import random
trainval_percent = 0.1
train_percent = 0.9
xmlfilepath = 'mycoco/VOC2028/JPEGImages'
txtsavepath = 'mycoco/VOC2028/ImageSets'
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv) #从所有list中返回tv个数量的项目
train = random.sample(trainval, tr)
if not os.path.exists('ImageSets/'):
    os.makedirs('ImageSets/')
ftrainval = open('mycoco/VOC2028/ImageSets/trainval.txt', 'w')
ftest = open('mycoco/VOC2028/ImageSets/test.txt', 'w')
ftrain = open('mycoco/VOC2028/ImageSets/train.txt', 'w')
fval = open('mycoco/VOC2028/ImageSets/val.txt', 'w')
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
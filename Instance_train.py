import pandas as pd
import torch
import os
from model import *
import numpy as np

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()


def trainbinaryunet3d():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('/home/zhanglang/python_product/COVID19/dataprocess/data1/traindata.csv')
    maskdatasource = csvdata.iloc[:, 1].values           #取所有行第二列数据(label)(处理表格数据)
    imagedatasource = csvdata.iloc[:, 0].values          #取所有行第一列数据（image)

    csvdataaug = pd.read_csv('/home/zhanglang/python_product/COVID19/dataprocess/data1/trainaugdata.csv')
    maskdataaug = csvdataaug.iloc[:, 1].values
    imagedataaug = csvdataaug.iloc[:, 0].values

    imagedata = np.concatenate((imagedatasource, imagedataaug), axis=0) #原图像和数据增强后的图像 融合
    maskdata = np.concatenate((maskdatasource, maskdataaug), axis=0)    #原图像和数据增强后的图像 融合

    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    data_dir2 = '/home/zhanglang/python_product/COVID19/dataprocess/data1/validata.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet3d = BinaryUNet3dModel(image_depth=32, image_height=320, image_width=256, image_channel=1, numclass=1,
                               batch_size=1, loss_name='BinaryDiceLoss')
    unet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/instance/dice/Unet',
                        epochs=1, showwind=[4, 8])


def trainbinaryVnet3d():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('/home/zhanglang/python_product/COVID19/dataprocess/data1/traindata.csv')
    maskdatasource = csvdata.iloc[:, 1].values
    imagedatasource = csvdata.iloc[:, 0].values
    csvdataaug = pd.read_csv('/home/zhanglang/python_product/COVID19/dataprocess/data1/trainaugdata.csv')
    maskdataaug = csvdataaug.iloc[:, 1].values
    imagedataaug = csvdataaug.iloc[:, 0].values
    imagedata = np.concatenate((imagedatasource, imagedataaug), axis=0)
    maskdata = np.concatenate((maskdatasource, maskdataaug), axis=0)
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    data_dir2 = '/home/zhanglang/python_product/COVID19/dataprocess/data1/validata.csv'
    csv_data2 = pd.read_csv(data_dir2)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    vnet3d = BinaryVNet3dModel(image_depth=32, image_height=320, image_width=256, image_channel=1, numclass=1,
                               batch_size=1, loss_name='BinaryDiceLoss')
    vnet3d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/instance/dice/Vnet',
                        epochs=2, showwind=[4, 8])


if __name__ == '__main__':
    trainbinaryunet3d()
    # trainbinaryVnet3d()

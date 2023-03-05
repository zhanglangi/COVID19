import torch
import os
from model import *
from dataprocess.utils import file_name_path, MorphologicalOperation, GetLargestConnectedCompont, \
    GetLargestConnectedCompontBoundingbox
import SimpleITK as sitk
import numpy as np

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()


def inferencemutilunet3dtest():
    newSize = (256, 320, 32)
    ##创建一个 BinaryUNet3dModel 的实例，用于进行3D Unet模型的分割操作。
    Unet3d = BinaryUNet3dModel(image_depth=32, image_height=320, image_width=256, image_channel=1, numclass=1,
                               batch_size=1, loss_name='BinaryDiceLoss', inference=True,
                               model_path=r'log\instance\dice\Unet\BinaryUNet3d.pth')
    ##读取CT图像文件夹的路径和保存分割结果的文件夹路径。
    datapath = r"D:\Program_files\python_project\COVID19\lung_COVID19\20_ncov_scan"
    makspath = r"D:\Program_files\python_project\COVID19\log\predict\Unet"
    image_path_list = file_name_path(datapath, False, True)
    ##对于每一个CT图像，读取其对应的 .nii 文件
    ##并进行预处理操作。预处理操作包括二值化、形态学操作、取最大联通区域等操作，以提取出感兴趣的血管区域。
    for i in range(len(image_path_list)):
        imagepathname = datapath + "/" + image_path_list[i]
        src = sitk.ReadImage(imagepathname)
        ##二值化、形态学操作、取最大联通区域
        binary_src = sitk.BinaryThreshold(src, 100, 5000)
        binary_src = MorphologicalOperation(binary_src, 1)
        binary_src = GetLargestConnectedCompont(binary_src)
        boundingbox = GetLargestConnectedCompontBoundingbox(binary_src)
        # print(boundingbox)  # (x,y,z,xlength,ylength,zlength)
        x1, y1, z1, x2, y2, z2 = boundingbox[0], boundingbox[1], boundingbox[2], boundingbox[0] + boundingbox[3], \
                                 boundingbox[1] + boundingbox[4], boundingbox[2] + boundingbox[5]
        src_array = sitk.GetArrayFromImage(src)                       #将模型预测的二进制掩模（src）原始数据体积相同的二进制数组
        roi_src_array = src_array[z1:z2, y1:y2, x1:x2]                #
        roi_src = sitk.GetImageFromArray(roi_src_array)               #将二进制数组转换为SimpleITK图像格式
        roi_src.SetSpacing(src.GetSpacing())                          #为新生成的二进制掩模图像（roi_src）设置相同的空间坐标
        roi_src.SetDirection(src.GetDirection())                      #...方向
        roi_src.SetOrigin(src.GetOrigin())                            #...原点属性

        ##对预处理后的感兴趣区域进行 3D Unet 分割，得到血管分割结果。
        sitk_mask = Unet3d.inference(roi_src, newSize)

        ##
        roi_binary_array = sitk.GetArrayFromImage(sitk_mask)            #将模型预测的二进制掩模（sitk_mask）原始数据体积相同的二进制数组
        binary_array = np.zeros_like(src_array)                         #将roi_binary_array赋值给名为binary_array的全零数组的指定区域（即用boundingbox选出的感兴趣区域）
        binary_array[z1:z2, y1:y2, x1:x2] = roi_binary_array[:, :, :]   #将roi_binary_array赋值给名为binary_array的全零数组的指定区域（即用boundingbox选出的感兴趣区域）
        binary_vessels = sitk.GetImageFromArray(binary_array)           #将二进制数组转换为SimpleITK图像格式
        binary_vessels.SetSpacing(src.GetSpacing())                     #为新生成的二进制掩模图像（binary_vessels）设置相同的空间坐标
        binary_vessels.SetDirection(src.GetDirection())                 #....方向
        binary_vessels.SetOrigin(src.GetOrigin())                       #....原点属性

        ##将血管分割结果保存到指定文件夹中，文件名与对应的CT图像文件名相同。
        maskpathname = makspath + "/" + image_path_list[i]
        sitk.WriteImage(binary_vessels, maskpathname)


def inferencemutilvnet3dtest():
    newSize = (256, 320, 32)
    vnet3d = BinaryVNet3dModel(image_depth=32, image_height=320, image_width=256, image_channel=1, numclass=1,
                               batch_size=1, loss_name='BinaryDiceLoss', inference=True,
                               model_path=r'log\instance\dice\Unet\BinaryUNet3d.pth')
    datapath = r"D:\Program_files\python_project\COVID19\lung_COVID19\20_ncov_scan"
    makspath = r"D:\Program_files\python_project\COVID19\log\predict\Unet"
    image_path_list = file_name_path(datapath, False, True)
    for i in range(len(image_path_list)):
        imagepathname = datapath + "/" + image_path_list[i]
        src = sitk.ReadImage(imagepathname)
        binary_src = sitk.BinaryThreshold(src, 100, 5000)
        binary_src = MorphologicalOperation(binary_src, 1)
        binary_src = GetLargestConnectedCompont(binary_src)
        boundingbox = GetLargestConnectedCompontBoundingbox(binary_src)
        # print(boundingbox)  # (x,y,z,xlength,ylength,zlength)
        x1, y1, z1, x2, y2, z2 = boundingbox[0], boundingbox[1], boundingbox[2], boundingbox[0] + boundingbox[3], \
                                 boundingbox[1] + boundingbox[4], boundingbox[2] + boundingbox[5]
        src_array = sitk.GetArrayFromImage(src)
        roi_src_array = src_array[z1:z2, y1:y2, x1:x2]
        roi_src = sitk.GetImageFromArray(roi_src_array)
        roi_src.SetSpacing(src.GetSpacing())
        roi_src.SetDirection(src.GetDirection())
        roi_src.SetOrigin(src.GetOrigin())

        sitk_mask = vnet3d.inference(roi_src, newSize)

        roi_binary_array = sitk.GetArrayFromImage(sitk_mask)
        binary_array = np.zeros_like(src_array)
        binary_array[z1:z2, y1:y2, x1:x2] = roi_binary_array[:, :, :]
        binary_vessels = sitk.GetImageFromArray(binary_array)
        binary_vessels.SetSpacing(src.GetSpacing())
        binary_vessels.SetDirection(src.GetDirection())
        binary_vessels.SetOrigin(src.GetOrigin())

        maskpathname = makspath + "/" + image_path_list[i]
        sitk.WriteImage(binary_vessels, maskpathname)


if __name__ == '__main__':
    # inferencemutilunet3dtest()
    inferencemutilvnet3dtest()

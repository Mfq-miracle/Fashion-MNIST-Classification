# 数据增强方法
import cv2
import numpy as np
import shutil
import os
import pathlib


def gauss_noise(img, path_out_gauss, mean=0, var=0.001):
    """
        添加高斯噪声
        mean : 均值
        var : 方差
    """
    img = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out = img + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    cv2.imwrite(path_out_gauss, out)


def mirror(img, path_out_mirror):
    """
        水平镜像
    """
    h_flip = cv2.flip(img, 1)
    cv2.imwrite(path_out_mirror, h_flip)


def rotate(img, path_out_rotate):
    """
        旋转
    """
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imwrite(path_out_rotate, dst)


def shear(img, path_out_shear):
    """
        剪切
    """
    height, width = img.shape[:2]
    cropped = img[int(height / 9):height, int(width / 9):width]
    cv2.imwrite(path_out_shear, cropped)


# 对输入路径文件夹内的所有图片进行数据增广，输出到目标路径
def image_aug(image_path, image_out_path):
    if not os.path.exists(image_out_path):
        os.mkdir(image_out_path)
    img_list = os.listdir(image_path)

    print("\n")
    print("----------------------------------------")
    print("The original data path:" + image_path)
    print("The original data set size:" + str(len(img_list)))
    print("----------------------------------------")

    imageNameList = [
        '_gasuss.jpg',
        '_mirror.jpg',
        # '_rotate.jpg',
        # '_shear.jpg',
        '.jpg']

    for i in range(0, len(img_list)):
        path = os.path.join(image_path, img_list[i])
        out_image_name = os.path.splitext(img_list[i])[0]
        for j in range(0, len(imageNameList)):
            path_out = os.path.join(
                image_out_path, out_image_name + imageNameList[j])
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if j == 0:
                gauss_noise(image, path_out)
            elif j == 1:
                mirror(image, path_out)
            # elif j == 2:
            #     rotate(image, path_out)
            # elif j == 3:
            #     shear(image, path_out)
            else:
                shutil.copy(path, path_out)

    print("----------------------------------------")
    print("The data augmention path:" + image_out_path)
    outlist = os.listdir(image_out_path)
    print("The data augmention sizes:" + str(len(outlist)))
    print("----------------------------------------")
    print("Rich sample for:" + str(len(outlist) - len(img_list)))


# 对数据集训练集目录下的各类别数据运用image_aug
def data_aug(src_path, out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    dir_list = os.listdir(src_path)
    for i in range(0, len(dir_list)):
        src = os.path.join(src_path, dir_list[i])
        out = os.path.join(out_path, dir_list[i])
        image_aug(src, out)


# 测试
# data_path = './images/fashion_mnist/train'
# data_out_path = './images/fashion_mnist/aug_train'
# data_aug(data_path, data_out_path)

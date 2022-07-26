# 数据生成器函数
import struct
import numpy as np
import tensorflow as tf
import os
from image_aug import data_aug
import numpy as np
import cv2
import pathlib
AUTOTUNE = tf.data.experimental.AUTOTUNE


# 用于读取idx类型数据集,如MNIST、Fashion Mnist
def decode_idx_ubyte(idx_ubyte_file):
    # 读取二进制数据
    bin_data = open(idx_ubyte_file, 'rb').read()

    """
    struct.unpack_from方法有三个参数，分别是解析格式、二进制流、开始解析位置
    其中解析格式又分为两部分，打头的>表示高位优先，对应地<表示低位优先；
    接下来的不定长字符串表示数据类型，比如x 仅占位、c 1位字符、B 1位无符号整数、
    i 4位有符号整数、I 4位无符号整数、f 4位小数、d 8位小数、s 字符串；每种格式
    前面还可以加数字表示个数，比如4i和iiii等价
    """
    # 先定义IDX第三位数据类型与struct中格式字符的映射关系
    data_types = {
        8: 'B',  # 1位无符号整数
        9: 'b',  # 1位有符号整数
        11: 'h',  # 2位有符号整数
        12: 'i',  # 4位有符号整数
        13: 'f',  # 4位小数
        14: 'd'  # 8位小数
    }

    # 解析文件头信息
    fmt_magic = ">2x2B"
    offset = 0
    data_type, dim = struct.unpack_from(fmt_magic, bin_data, offset)

    fmt_dim = ">" + str(dim) + "i"
    offset = offset + struct.calcsize(fmt_magic)
    dim_list = struct.unpack_from(fmt_dim, bin_data, offset)

    # 计算总读取长度需要把dim_list的几个维数乘起来
    # 这里可以用reduce方法，第一个参数为俩输入变量的函数，函数返回结果后
    # 再从列表里取第三个值然后送给函数再算，再取第四个值。。。
    from functools import reduce
    all_size = reduce(lambda x1, x2: x1 * x2, dim_list)
    fmt_all = ">" + str(all_size) + data_types[data_type]
    offset = offset + struct.calcsize(fmt_dim)
    data = struct.unpack_from(fmt_all, bin_data, offset)
    data_set = np.array(data).reshape(dim_list)
    return data_set


# idx格式fashion mnist 数据加载
def fashion_mnist_dataloader(path):
    train_images = decode_idx_ubyte(path + "//train-images-idx3-ubyte")
    train_labels = decode_idx_ubyte(path + "//train-labels-idx1-ubyte")
    test_images = decode_idx_ubyte(path + "//t10k-images-idx3-ubyte")
    test_labels = decode_idx_ubyte(path + "//t10k-labels-idx1-ubyte")
    return (train_images, train_labels), (test_images, test_labels)


# 接上，针对idx格式fashion mnist的数据生成器
def data_generator(path, BATCH_SIZE=32, transformer=False):
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist_dataloader(path)

    # 数据增强测试
    # aug_images = train_images
    # train_images = np.expand_dims(train_images, axis=3)
    # cv2.imwrite("./images/res_test.jpg", train_images[0])

    # for item in train_images[:10000]:
    #     temp = contrast_brightness_image(item, 1.2, 10)
    #     temp = np.expand_dims(temp, axis=0)
    #     aug_images = np.vstack((aug_images, temp))

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images_ds = tf.data.Dataset.from_tensor_slices(train_images)
    train_labels_ds = tf.data.Dataset.from_tensor_slices(train_labels)
    train_ds = tf.data.Dataset.zip((train_images_ds, train_labels_ds))
    train_ds = train_ds.shuffle(buffer_size=10000)
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, test_images, test_labels


# 针对fashion mnist jpg格式数据集的数据生成器
def fashion_mnist_dataloader_jpg(path, batch_size=32, transformer=False):
    data_dir = pathlib.Path(path)
    train_dir = os.path.join(data_dir, "train")
    aug_dir = os.path.join(data_dir, "aug_train")
    val_dir = os.path.join(data_dir, "test")
    if transformer:
        data_aug(train_dir, aug_dir)
        train_dir = aug_dir
    # image_count = len(list(data_dir.glob('*/*.jpg')))

    img_height = 28
    img_width = 28

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode="grayscale"
    )

    return train_ds, val_ds


# 测试
# data_dir = "C://Users//97546//.keras//datasets//fashion-mnist"
# BATCH_SIZE = 64
# train_ds, test_images, test_labels = data_generator(data_dir, BATCH_SIZE=BATCH_SIZE)


# 将fashion mnist转换为jpg格式
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist_dataloader(data_dir)
# for n in range(len(test_images)):
#     img = np.expand_dims(test_images[n], axis=2)
#     image_out_path = "./images/fashion_mnist/test/"+str(test_labels[n])
#     if not os.path.exists(image_out_path):
#         os.mkdir(image_out_path)
#     path_out = os.path.join(
#         image_out_path, str(n) + ".jpg")
#     cv2.imwrite(path_out, img)

# data_dir = "D://PycharmProject//FashionMNIST_StartProject//images//fashion_mnist"
# train_ds, val_ds = fashion_mnist_dataloader_jpg(data_dir)
#
#
# AUTOTUNE = tf.data.AUTOTUNE
#
# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#
# num_classes = 10
#
# model = tf.keras.Sequential([
#     tf.keras.layers.Rescaling(1./255),
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(32, 3, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(num_classes)
# ])
#
# model.compile(
#     optimizer='adam',
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=['accuracy'])
#
# model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=5
# )

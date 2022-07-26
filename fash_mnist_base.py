# fashion mnist 基本分类模型
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
AUTOTUNE = tf.data.experimental.AUTOTUNE

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 绘制数据集图片
# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# 归一化
train_images = train_images / 255.0
train_images = np.expand_dims(train_images, axis=-1)
test_images = test_images / 255.0
test_images = np.expand_dims(test_images, axis=-1)

train_images_ds = tf.data.Dataset.from_tensor_slices(train_images)
train_labels_ds = tf.data.Dataset.from_tensor_slices(train_labels)

image_label_ds = tf.data.Dataset.zip((train_images_ds, train_labels_ds))

BATCH_SIZE = 32

ds = image_label_ds.shuffle(buffer_size=60000)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)


# 验证数据
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# 构建模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# 模型编译
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

steps_per_epoch = tf.math.ceil(60000/BATCH_SIZE).numpy()
# 训练模型
model.fit(ds, epochs=5, steps_per_epoch=steps_per_epoch)
# 评估准确率
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)











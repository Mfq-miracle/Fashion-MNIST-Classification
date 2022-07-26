# 基于自定义数据生成器的fashion mnist分类模型
import tensorflow as tf
from tensorflow import keras
from fash_mnist_dataloader import fashion_mnist_dataloader_jpg
import cv2

data_path = "./images/fashion_mnist"
BATCH_SIZE = 64

train_ds, val_ds = fashion_mnist_dataloader_jpg(data_path,
                                                batch_size=BATCH_SIZE,
                                                transformer=True)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28, 1)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(10)
# ])

num_classes = 10

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(padding='same'),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(padding='same'),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
steps_per_epoch = train_ds.cardinality().numpy()
# steps_per_epoch = tf.math.ceil(120000/BATCH_SIZE).numpy()

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5,
          steps_per_epoch=steps_per_epoch)


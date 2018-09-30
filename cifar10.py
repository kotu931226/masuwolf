from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    EarlyStopping
)

import numpy as np
from models import modelset

lr_reducer = ReduceLROnPlateau(
    factor=np.sqrt(0.1),
    cooldown=0,
    patience=5,
    min_lr=0.5e-6
)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_cifar10.csv')

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = False

img_rows, img_cols = 32, 32
img_channels = 3

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

mean_image = np.mean(x_train, axis=0)
x_train -= mean_image
x_test -= mean_image
x_train /= 128.
x_test /= 128.

model = modelset.ResNet_API.build_res_18(
    (img_channels, img_cols, img_rows),
    nb_classes
)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

if not data_augmentation:
    model.fit(
        x_train,
        Y_train,
        batch_size=batch_size,
        nb_epoch=nb_epoch,
        validation_data=(x_test, Y_test),
        shuffle=True,
        callabacks=[lr_reducer, early_stopper, csv_logger]
    )

# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import (
#     ReduceLROnPlateau,
#     CSVLogger,
#     EarlyStopping
# )

import numpy as np
from sklearn.model_selection import train_test_split
from models import modelset_v3

# lr_reducer = ReduceLROnPlateau(
#     factor=np.sqrt(0.1),
#     cooldown=0,
#     patience=3,
#     min_lr=0.5e-6,
#     min_delta=0.1
# )
# early_stopper = EarlyStopping(min_delta=0.001, patience=10)
# csv_logger = CSVLogger('resnet18_cifar10.csv')

batch_size = 32
nb_epoch = 50

x = np.load('onehot_x.npy')
y = np.load('onehot_y.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

x_train = x_train.astype('float32').reshape(x_train.shape+(1,))
x_test = x_test.astype('float32').reshape(x_test.shape+(1,))

print(x_train.shape[1:])

resnet = modelset_v3.ResNet_API()
model = resnet.build(x_train.shape[1:])

model.summary()
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    validation_data=(x_test, y_test),
    shuffle=True,
    # callabacks=[lr_reducer, early_stopper, csv_logger]
)

# model.save("analogy_wolflog_20181008.h5")

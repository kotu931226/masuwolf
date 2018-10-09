import keras
from keras import backend as K
from keras import layers
from keras.layers import Conv2D, Activation, BatchNormalization
from keras.layers import Input, Add, Dense, LeakyReLU, GlobalAveragePooling2D
from keras.regularizers import l2


def ResNetConv2D(*args, **kwargs):
    conv_kwargs = {
        'padding':'same',
        'kernel_initializer':'he_normal',
        'kernel_regularizer':l2(1.e-4),
        'data_format':'channels_last'
    }
    conv_kwargs.update(kwargs)
    return Conv2D(*args, **conv_kwargs)

class ResNet_API:
    def __init__(self):
        pass

    def build(self, input_shape):
        inputs = Input(shape=input_shape)
        x = ResNetConv2D(filters=16, kernel_size=3)(inputs)
        x = self._res_block(x, 16)
        x = self._res_block(x, 16)
        x = self._res_block(x, 32, 2)
        x = self._res_block(x, 32)
        x = self._res_block(x, 32)
        x = self._res_block(x, 64, 2)
        x = self._res_block(x, 64)
        x = self._res_block(x, 64)
        x = self._res_block(x, 128, 2)
        x = self._res_block(x, 128)
        x = self._res_block(x, 128)
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=5, kernel_initializer='he_normal')(x)
        prediction = Activation('softmax')(x)
        return keras.Model(inputs=inputs, outputs=prediction)


    def _res_block(self, x, filters, strides=1):
        dx = LeakyReLU(alpha=0.1)(x)
        dx = BatchNormalization(axis=-1)(dx)
        dx = ResNetConv2D(filters, 3, strides=strides)(dx)
        dx = BatchNormalization(axis=-1)(dx)
        dx = LeakyReLU(alpha=0.1)(dx)
        dx = layers.Dropout(0.5)(dx)
        dx = ResNetConv2D(filters, 3)(dx)
        return self.shortcut(x, dx)

    def shortcut(self, x, dx):
        x_shape = K.int_shape(x)
        dx_shape = K.int_shape(dx)

        if x_shape == dx_shape:
            shortcut_x = x
        else:
            stride_w = int(round(x_shape[1] / dx_shape[1]))
            stride_h = int(round(x_shape[2] / dx_shape[2]))

            shortcut_x = Conv2D(
                filters=dx_shape[3],
                kernel_size=(1, 1),
                strides=(stride_w, stride_h),
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1.e-4)
            )(x)
        return Add()([shortcut_x, dx])
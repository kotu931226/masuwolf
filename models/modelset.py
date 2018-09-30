from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    GlobalAveragePooling2D,
    Add,
    BatchNormalization,
    LeakyReLU,
)
# from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def ResNetConv2D(*args, **kwargs):
    conv_kwargs = {
        'strides':(1, 1),
        'padding':'same',
        'kernel_initializer':'he_normal',
        'kernel_regularizer':l2(1.e-4)
    }
    conv_kwargs.update(kwargs)
    return Conv2D(*args, **kwargs)

def bn_relu_conv(*args, **kwargs):
    def f(x):
        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU(alpha=0.1)(x)
        return ResNetConv2D(*args, **kwargs)
    return f




def shortcut(dx, x):
    x_shape = K.int_shape(x)
    dx_shape = K.int_shape(dx)

    if x_shape == dx_shape:
        shortcut_x = x
    else:
        stride_w = int(round(dx_shape[1]/x_shape[1]))
        stride_h = int(round(dx_shape[2]/x_shape[2]))

        shortcut_x = Conv2D(
            filters=x_shape[3],
            kernel_size=(1, 1),
            strides=(stride_w, stride_h),
            kernel_initializer='he_normal',
            kernel_regularizer=l2(1.e-4)
        )(dx)
    return Add()([shortcut_x, x])

def basic_block(filters, first_strides):
    def f(x):
        conv1 = bn_relu_conv(filters=filters, kernel_size=(3, 3), strides=first_strides)(x)
        conv2 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return shortcut(conv2, x)
    return f

def bottleneck_block(filters, first_strides):
    def f(x):
        conv1 = bn_relu_conv(filters=filters, kernel_size=(1, 1), strides=first_strides)(x)
        conv2 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        conv3 = bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv2)
        return shortcut(conv3, x)
    return f

def resdiual_blocks(block_func, filters, repetitions):
    def f(x):
        for i in range(repetitions):
            first_strides = (2, 2) if i == 0 else (1, 1)

            x = block_func(filters=filters, first_strides=first_strides)(x)
        return x
    return f

class ResNet_API():
    @staticmethod
    def build(input_shape, num_class, repetitions, filters=64, is_bottleneck=False):
        if not is_bottleneck:
            block_func = basic_block
        else:
            block_func = bottleneck_block
        
        input_x = Input(shape=input_shape)
        block = input_x
        for r in repetitions:
            block = resdiual_blocks(block_func=block_func, filters=filters, repetitions=r)(block)
            filters *= 2
        pool = GlobalAveragePooling2D()(block)
        fc1 = Dense(units=num_class, kernel_initializer='he_normal', activation='softmax')(pool)

        return Model(inputs=input_x, outputs=fc1)

    @staticmethod
    def build_res_wolf_96x16(input_shape, num_class):
        return ResNet_API.build(input_shape, num_class, [4, 4])

    @staticmethod
    def build_res_18(input_shape, num_class):
        return ResNet_API.build(input_shape, num_class, [2, 2, 2, 2])

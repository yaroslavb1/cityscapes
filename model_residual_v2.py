from tensorflow.keras.layers import Input, Add, Concatenate, Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.models import Model, Sequential
from depthwise_context_conv import DepthwiseContextConv2D


def create_model(input_shape=(None, None, 3), depths=[1, 2, 4, 8], width=32, depth_multiplier=4, max_drs=[7, 7, 7, 5], activation='swish'):
    x = inputs = Input(input_shape)

    ds = []
    widths = []
    for depth, max_dr in zip(depths, max_drs):
        x = Conv2D(width, 3, strides=2, use_bias=False, padding='same')(x)
        x = SyncBatchNormalization()(x)
        x = skip = Activation(activation)(x)
        for _ in range(depth):
            x = Conv2D(width, 1, padding='same')(x)
            x = SyncBatchNormalization()(x)
            x = Activation(activation)(x)
            x = DepthwiseContextConv2D(depth_multiplier=depth_multiplier, max_learned_dilation_rate=max_dr)(x)
            x = SyncBatchNormalization()(x)
            x = Activation(activation)(x)
            x = Conv2D(width, 1, padding='same')(x)
            x = SyncBatchNormalization()(x)
            x = Activation(activation)(x)
            x = skip = Add()([x, skip])
        ds.append(x)
        widths.append(width)
        width *= 2
    
    for y, width in zip(reversed(ds[:-1]), reversed(widths[:-1])):
        x = UpSampling2D(size=2, interpolation='bilinear')(x)
        x = Concatenate()([x, y])
        x = Conv2D(width, 1)(x)
        x = SyncBatchNormalization()(x)
        x = Activation(activation)(x)

    x = Conv2D(19, 1, padding='same', activation='softmax')(x)
    x = UpSampling2D(size=2, interpolation='bilinear')(x)

    model = Model(inputs=inputs, outputs=x)
        
    return model

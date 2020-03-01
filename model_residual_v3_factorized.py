from tensorflow.keras.layers import Input, Add, Concatenate, Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import VarianceScaling
from depthwise_context_conv import FastDepthwiseContextConv2D


def create_model(input_shape=(None, None, 3), depths=[1, 2, 4, 8], width=32, sqrt_depth_multiplier=2, max_drs=[7, 7, 7, 5], activation='relu'):
    x = inputs = Input(input_shape)

    ds = []
    widths = []
    for depth, max_dr in zip(depths, max_drs):
        x = Conv2D(width, 3, strides=2, use_bias=False, padding='same', kernel_initializer='he_uniform')(x)
        x = skip = Activation(activation)(x)
        for _ in range(depth):
            x = Conv2D(width, 1, padding='same', kernel_initializer='he_uniform')(x)
            x = Activation(activation)(x)
            x = FastDepthwiseContextConv2D(sqrt_depth_multiplier=sqrt_depth_multiplier, max_learned_dilation_rate=max_dr, kernel_initializer='he_uniform')(x)
            x = Activation(activation)(x)
            # x = Conv2D(width, 1, padding='same', kernel_initializer=VarianceScaling(2 / sum(depths), mode='fan_in', distribution='uniform'))(x)
            x = Conv2D(width, 1, padding='same', kernel_initializer='zeros')(x)
            x = Activation(activation)(x)
            x = skip = Add()([x, skip])
        ds.append(x)
        widths.append(width)
        width *= 2
    
    for y, width in zip(reversed(ds[:-1]), reversed(widths[:-1])):
        x = UpSampling2D(size=2, interpolation='bilinear')(x)
        x = Concatenate()([x, y])
        x = Conv2D(width, 1, kernel_initializer='he_uniform')(x)
        x = Activation(activation)(x)

    # x = Conv2D(19, 1, padding='same', activation='softmax', kernel_initializer='he_uniform')(x)
    x = Conv2D(19, 1, padding='same', activation='softmax', kernel_initializer='zeros')(x)
    x = UpSampling2D(size=2, interpolation='bilinear')(x)

    model = Model(inputs=inputs, outputs=x)
        
    return model

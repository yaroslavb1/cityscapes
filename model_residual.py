from tensorflow.keras.layers import Input, Add, Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.models import Model, Sequential
from depthwise_context_conv import DepthwiseContextConv2D


def create_model(input_shape=(None, None, 3), depth=6, width=32, depth_multiplier=4):
    x = inputs = Input(input_shape)

    # stem
    x = Conv2D(width, 3, strides=2, use_bias=False, padding='same', input_shape=input_shape)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(width*2, 3, strides=2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(width*4, 3, strides=2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = skip = Activation('relu')(x)
    
    for i in range(depth):
        x = Conv2D(width*4, 1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = DepthwiseContextConv2D(depth_multiplier=depth_multiplier)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(width*4, 1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = skip = Add()([x, skip])

    x = Conv2D(19, 1, padding='same', activation='softmax')(x)
    x = UpSampling2D(size=8, interpolation='bilinear')(x)

    model = Model(inputs=inputs, outputs=x)
        
    return model

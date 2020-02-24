from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.models import Model, Sequential
from depthwise_context_conv import DepthwiseContextConv2D


def create_model(input_shape=(None, None, 3)):
    layers = []
    # stem
    layers += [
        Conv2D(32, 3, strides=2, use_bias=False, padding='same', activation='relu', input_shape=input_shape),
        Conv2D(64, 3, strides=2, use_bias=False, padding='same', activation='relu'),
        Conv2D(128, 3, strides=2, use_bias=False, padding='same', activation='relu'),
    ]
    for i in range(12):
        layers += [
            Conv2D(128, 1, padding='same', activation='relu'),
            DepthwiseContextConv2D(depth_multiplier=4, activation='relu'),
            Conv2D(128, 1, padding='same', activation='relu'),
        ]
    layers += [
        Conv2D(19, 1, padding='same', activation='softmax'),
        UpSampling2D(size=8, interpolation='bilinear'),
    ]
    model = Sequential(layers)
    return model

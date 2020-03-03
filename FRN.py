from functools import reduce

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


class FilterResponseNormalization(keras.layers.Layer):
    def __init__(self, activation='swish', **kwargs):
        self.activation = keras.activations.get(activation)
        super(FilterResponseNormalization, self).__init__(**kwargs)
        
    def build(self, input_shape):
        assert len(input_shape) == 4
        C = input_shape[-1]
        N = reduce(lambda x, y: x * y, input_shape[1:-1], 1)
        if N == 1:
            self.epsilon = self.add_weight(shape=(1,), name='FRN_epsilon', initializer=keras.initializers.Constant(1), constraint=lambda e: K.maximum(e, 1e-6))
        else:
            self.epsilon = 1e-6
        self.gamma = self.add_weight(shape=(1, 1, 1, C), name='FRN_gamma')
        self.beta = self.add_weight(shape=(1, 1, 1, C), name='FRN_beta')
        self.tau = self.add_weight(shape=(1, 1, 1, C), name='FRN_tau')
        self.built = True

    def call(self, inputs):
        x = inputs
        nu2 = K.mean(x**2, axis=[1, 2], keepdims=True)
        x = x / K.sqrt(nu2 + self.epsilon)
        x = self.gamma * x + self.beta
        x = self.activation(x) + self.tau
        return x

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 4
        return input_shape

    def get_config(self):
        return {'activation': keras.activations.serialize(self.activation)}

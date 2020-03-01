import tensorflow as tf
import tensorflow.keras.backend as K
from keras.utils import conv_utils
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.python.keras.engine.input_spec import InputSpec


# Could implement bilinear mode more efficiently by padding 1 extra, doing a few 2x2 convs with bilinear 
# interpolation coefficients, and then doing an integer-dilated 4x4 conv.
# Can do a similar trick for gaussian, by gaussian blurring instead of the 2x2 bilinear conv.
# The following should deal with non-differentiability issues:
# dr_int = stopgrad(floor(dr))  # no grad
# dr_frac = dr - dr_int         # grad


class DepthwiseContextConv2D(tf.keras.layers.Conv2D):
    def __init__(
        self, 
        kernel_size=3,
        strides=1,
        depth_multiplier=1,
        activation=None, 
        max_learned_dilation_rate=7,
        sigma=1,
        pad_mode='REFLECT',
        use_bias=True,
        depthwise_initializer='he_normal',
        bias_initializer='zeros',
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(DepthwiseContextConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            activation=activation,
            use_bias=use_bias,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        self.depth_multiplier = depth_multiplier
        self.max_dr = conv_utils.normalize_tuple(max_learned_dilation_rate, 2, 'max_learned_dilation_rate')
        self.sigma = conv_utils.normalize_tuple(sigma, 2, 'sigma')  # lower sigma is faster but less accurate dilation learning,
        self.pad_mode = pad_mode
        self.use_bias = use_bias
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

        assert pad_mode in ('REFLECT', 'CONSTANT', 'SYMMETRIC')
        # These constraints are to not have to deal with padding/cropping/indexing issues.
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        assert self.max_dr[0] % 2 == 1
        assert self.max_dr[1] % 2 == 1
        
    def build(self, input_shape):
        assert len(input_shape) == 4
        if self.data_format == 'channels_first':
            channel_axis = 1
            self._data_format_for_dwconv = 'NCHW'
        else:
            channel_axis = 3
            self._data_format_for_dwconv = 'NHWC'
        input_dim = int(input_shape[channel_axis])
        
        kh, kw = self.kernel_size
        max_drh, max_drw = self.max_dr

        depthwise_kernel_shape = (kh, kw, input_dim, self.depth_multiplier)
        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            name='depthwise_kernel',
            initializer=self.depthwise_initializer,
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(input_dim * self.depth_multiplier,),
                name='bias',
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
            
        self.dilation_rate_h = self.add_weight(
            shape=(1, 1, input_dim, self.depth_multiplier),
            name='dilation_rate_h',
            constraint=constraints.MinMaxNorm(1, max_drh, axis=[]),
            initializer=initializers.RandomUniform(1, max_drh),
            # regularizer=None,
        )
        self.dilation_rate_w = self.add_weight(
            shape=(1, 1, input_dim, self.depth_multiplier),
            name='dilation_rate_w',
            constraint=constraints.MinMaxNorm(1, max_drw, axis=[]),
            initializer=initializers.RandomUniform(1, max_drw),
            # regularizer=None,
        )
        
        padh = kh * max_drh
        padw = kw * max_drw
        padh0 = padh - padh // 2
        padh1 = padh // 2
        padw0 = padw - padw // 2
        padw1 = padw // 2
        self._paddings = [[0, 0], [padh0, padh1], [padw0, padw1], [0, 0]]
        self._xxh = tf.reshape(tf.range(-(kh-1)/2, (kh-1)/2+1), [1, -1, 1, 1])
        self._yyh = tf.reshape(tf.range(-padh/2, padh/2+1), [-1, 1, 1, 1])
        self._xxw = tf.reshape(tf.range(-(kw-1)/2, (kw-1)/2+1), [-1, 1, 1, 1])
        self._yyw = tf.reshape(tf.range(-padw/2, padw/2+1), [1, -1, 1, 1])
        self._strides_for_dwconv = [1, 1, self.strides[0], self.strides[1]]
        self._padding_for_dwconv = self.padding.upper()

        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True
    
    def call(self, inputs):
        x = inputs
        x = tf.pad(x, self._paddings, mode=self.pad_mode)
        x = tf.cumsum(tf.cumsum(x, axis=1), axis=2)
        x = tf.nn.depthwise_conv2d(
            x, 
            self._dilated_depthwise_kernel, 
            self._strides_for_dwconv, 
            padding=self._padding_for_dwconv, 
            data_format=self._data_format_for_dwconv,
        )
        if self.use_bias:
            x = K.bias_add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
    @property
    def _dilated_depthwise_kernel(self):
        left = self._calculate_dilation_matrix(self._xxh, self._yyh, self.dilation_rate_h, self.sigma[0], 0)
        right = self._calculate_dilation_matrix(self._xxw, self._yyw, self.dilation_rate_w, self.sigma[1], 1)
        tmp = tf.einsum('ab...,bc...->ac...', left, self.depthwise_kernel)
        return tf.einsum('ab...,bc...->ac...', tmp, right)
    
    @staticmethod
    def _calculate_dilation_matrix(xx, yy, d, sigma=1, norm_axis=None):
        if False:
            # bilinear
            delta_to_pos_1 = yy - (xx-0.5)*d
            delta_to_neg_1 = yy - (xx+0.5)*d
            mat_pos = K.maximum(1 - K.abs(delta_to_pos_1), 0)
            mat_neg = K.maximum(1 - K.abs(delta_to_neg_1), 0)
        else:
            # gaussian
            relative_delta_to_pos_1 = yy/d - (xx-0.5)
            relative_delta_to_neg_1 = yy/d - (xx+0.5)
            mat_pos = K.exp(-(relative_delta_to_pos_1 / sigma)**2)
            mat_pos = mat_pos / K.sum(mat_pos, axis=norm_axis, keepdims=True)
            mat_neg = K.exp(-(relative_delta_to_neg_1 / sigma)**2)
            mat_neg = mat_neg / K.sum(mat_neg, axis=norm_axis, keepdims=True)
        mat = mat_pos - mat_neg
        # normalize weight matrix by area summed - output of contextconv will be weighted averagepools
        mat = mat / d
        return mat
    

# Factorize the big depthwise conv (which is currently super slow in training) into just x- and y- convs.
class FastDepthwiseContextConv2D(tf.keras.layers.Conv2D):
    def __init__(
        self, 
        kernel_size=3,
        strides=1,
        sqrt_depth_multiplier=1,
        activation=None, 
        max_learned_dilation_rate=7,
        sigma=1,
        pad_mode='REFLECT',
        use_bias=True,
        depthwise_initializer='he_normal',
        bias_initializer='zeros',
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super(FastDepthwiseContextConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            activation=activation,
            use_bias=use_bias,
            activity_regularizer=activity_regularizer,
            bias_regularizer=bias_regularizer,
            bias_constraint=bias_constraint,
            **kwargs,
        )

        self.sqrt_depth_multiplier = sqrt_depth_multiplier
        self.max_dr = conv_utils.normalize_tuple(max_learned_dilation_rate, 2, 'max_learned_dilation_rate')
        self.sigma = conv_utils.normalize_tuple(sigma, 2, 'sigma')  # lower sigma is faster but less accurate dilation learning,
        self.pad_mode = pad_mode
        self.use_bias = use_bias
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

        assert pad_mode in ('REFLECT', 'CONSTANT', 'SYMMETRIC')
        # These constraints are to not have to deal with padding/cropping/indexing issues.
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        assert self.max_dr[0] % 2 == 1
        assert self.max_dr[1] % 2 == 1
        
    def build(self, input_shape):
        assert len(input_shape) == 4
        if self.data_format == 'channels_first':
            channel_axis = 1
            self._data_format_for_dwconv = 'NCHW'
        else:
            channel_axis = 3
            self._data_format_for_dwconv = 'NHWC'
        input_dim = int(input_shape[channel_axis])
        
        kh, kw = self.kernel_size
        max_drh, max_drw = self.max_dr

        depthwise_kernel_h_shape = (kh, 1, input_dim, self.sqrt_depth_multiplier)
        self.depthwise_kernel_h = self.add_weight(
            shape=depthwise_kernel_h_shape,
            name='depthwise_kernel_h',
            initializer=self.depthwise_initializer,
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint,
        )
        depthwise_kernel_w_shape = (1, kw, input_dim * self.sqrt_depth_multiplier, self.sqrt_depth_multiplier)
        self.depthwise_kernel_w = self.add_weight(
            shape=depthwise_kernel_w_shape,
            name='depthwise_kernel_w',
            initializer=self.depthwise_initializer,
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(input_dim * self.sqrt_depth_multiplier ** 2,),
                name='bias',
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
            
        self.dilation_rate_h = self.add_weight(
            shape=(1, 1, input_dim, self.sqrt_depth_multiplier),
            name='dilation_rate_h',
            constraint=constraints.MinMaxNorm(1, max_drh, axis=[]),
            initializer=initializers.RandomUniform(1, max_drh),
            # regularizer=None,
        )
        self.dilation_rate_w = self.add_weight(
            shape=(1, 1, input_dim * self.sqrt_depth_multiplier, self.sqrt_depth_multiplier),
            name='dilation_rate_w',
            constraint=constraints.MinMaxNorm(1, max_drw, axis=[]),
            initializer=initializers.RandomUniform(1, max_drw),
            # regularizer=None,
        )
        
        padh = kh * max_drh
        padw = kw * max_drw
        padh0 = padh - padh // 2
        padh1 = padh // 2
        padw0 = padw - padw // 2
        padw1 = padw // 2
        self._paddings = [[0, 0], [padh0, padh1], [padw0, padw1], [0, 0]]
        self._xxh = tf.reshape(tf.range(-(kh-1)/2, (kh-1)/2+1), [1, -1, 1, 1])
        self._yyh = tf.reshape(tf.range(-padh/2, padh/2+1), [-1, 1, 1, 1])
        self._xxw = tf.reshape(tf.range(-(kw-1)/2, (kw-1)/2+1), [-1, 1, 1, 1])
        self._yyw = tf.reshape(tf.range(-padw/2, padw/2+1), [1, -1, 1, 1])
        self._strides_for_dwconv = [1, 1, self.strides[0], self.strides[1]]
        self._padding_for_dwconv = self.padding.upper()

        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True
    
    def call(self, inputs):
        x = inputs
        x = tf.pad(x, self._paddings, mode=self.pad_mode)
        x = tf.cumsum(tf.cumsum(x, axis=1), axis=2)
        x = tf.nn.depthwise_conv2d(
            x, 
            self._dilated_depthwise_kernel_h, 
            self._strides_for_dwconv, 
            padding=self._padding_for_dwconv, 
            data_format=self._data_format_for_dwconv,
        )
        x = tf.nn.depthwise_conv2d(
            x, 
            self._dilated_depthwise_kernel_w, 
            [1, 1, 1, 1], 
            padding=self._padding_for_dwconv, 
            data_format=self._data_format_for_dwconv,
        )
        if self.use_bias:
            x = K.bias_add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
    @property
    def _dilated_depthwise_kernel_h(self):
        left = self._calculate_dilation_matrix(self._xxh, self._yyh, self.dilation_rate_h, self.sigma[0], 0)
        return tf.einsum('ab...,bc...->ac...', left, self.depthwise_kernel_h)
    
    @property
    def _dilated_depthwise_kernel_w(self):
        right = self._calculate_dilation_matrix(self._xxw, self._yyw, self.dilation_rate_w, self.sigma[1], 1)
        return tf.einsum('ab...,bc...->ac...', self.depthwise_kernel_w, right)
    
    @staticmethod
    def _calculate_dilation_matrix(xx, yy, d, sigma=1, norm_axis=None):
        if False:
            # bilinear
            delta_to_pos_1 = yy - (xx-0.5)*d
            delta_to_neg_1 = yy - (xx+0.5)*d
            mat_pos = K.maximum(1 - K.abs(delta_to_pos_1), 0)
            mat_neg = K.maximum(1 - K.abs(delta_to_neg_1), 0)
        else:
            # gaussian
            relative_delta_to_pos_1 = yy/d - (xx-0.5)
            relative_delta_to_neg_1 = yy/d - (xx+0.5)
            mat_pos = K.exp(-(relative_delta_to_pos_1 / sigma)**2)
            mat_pos = mat_pos / K.sum(mat_pos, axis=norm_axis, keepdims=True)
            mat_neg = K.exp(-(relative_delta_to_neg_1 / sigma)**2)
            mat_neg = mat_neg / K.sum(mat_neg, axis=norm_axis, keepdims=True)
        mat = mat_pos - mat_neg
        # normalize weight matrix by area summed - output of contextconv will be weighted averagepools
        mat = mat / d
        return mat
    

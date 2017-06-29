from __future__ import division
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import init_ops


def l1_loss(labels, logits):
    return slim.losses.absolute_difference(labels, logits)


def l2_loss(labels, logits):
    return tf.nn.l2_loss(tf.sub(labels, logits))


def mean_squared_error(labels, logits):
    return slim.losses.mean_squared_error(labels, logits)


def _is_sequence(seq):
  return isinstance(seq, list)


def create_variables(name,
                     shape,
                     initializer=tf.contrib.layers.xavier_initializer(),
                     weight_decay=1e-4,
                     trainable=True):
    """
    Function to create a variable.
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param weight_decay:.
    :param trainable:.
    layers.
    :return: The created variable
    """
    scope = tf.get_variable_scope()
    regularizer = (scope.reuse is True) and None or tf.contrib.layers.l2_regularizer(scale=weight_decay)
    with tf.variable_scope(scope, initializer=initializer, regularizer=regularizer):
        var = tf.get_variable(name, shape=shape, trainable=trainable)
    return var


def batch_norm(x, is_training=True, reuse=False, momentum=0.9, epsilon=1e-5, name=None):
    """
    Helper function to do batch normalziation
    :param x: a tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC` and the second dimension if `data_format` is
      `NCHW`.
    :param momentum: decay for the moving average. Reasonable values for `decay` are close
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc. Lower
      `decay` value (recommend trying `decay`=0.9) if model experiences reasonably
      good training performance but poor validation and/or test performance.
    :param epsilon: small float added to variance to avoid dividing by zero.
    :param is_training: whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    :param reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    :param name: Optional scope for `variable_scope`.
    :return: the 4D tensor after being normalized
    """

    return slim.layers.batch_norm(x,
                             decay=momentum,
                             epsilon=epsilon,
                             is_training=is_training,
                             updates_collections=None,
                             reuse=reuse,
                             scale=True,
                             scope=name)


def conv2d(x,
           kernel_size,
           num_outputs,
           filters=None,
           stride=1,
           padding='SAME',
           weight_decay=1e-4,
           reuse=False,
           trainable=True,
           addbias=False,
           name=None):
    """
    Helper function to do 2d convolution
    :param x: input 4D tensor.
    :param kernel_size: integer, the size of kernel, e.g. 3x3, 5x5.
    :param num_outputs: integer, the number of output filters.
    :param filters: input 4D tensor of filter. If filter is None,
        a variable created as [kernel_size, kernel_size, in_channels, out_channels].
    :param stride: a list of length 2 `[stride_height, stride_width]`.
        Can be an int if both strides are the same. Note that presently
        both strides must have the same value.
    :param padding: one of `"VALID"` or `"SAME"`.
    :param reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    :param padding: one of `"VALID"` or `"SAME"`.
    :param trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    :param addbias: this parameter is valid for filter is not None.
    :param name: Optional scope for `variable_scope`.
    :return:
    """
    if filters is None:
        return slim.conv2d(x,
                           num_outputs,
                           kernel_size,
                           stride=stride,
                           padding=padding,
                           activation_fn=None,
                           weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                           weights_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                           reuse=reuse,
                           trainable=trainable,
                           scope=name)
    else:
        out = tf.nn.conv2d(x,
                           filters,
                           strides=[1, stride, stride, 1],
                           padding=padding,
                           name=name)
        if addbias:
            out = slim.bias_add(out, reuse=reuse, trainable=trainable, scope=name)
        return out


def conv3d(x,
           kernel_size,
           num_outputs,
           depth_outputs,
           stride=[1] * 3,
           padding='SAME',
           weight_decay=1e-4,
           reuse=False,
           trainable=True,

           name=None):
    """
    Helper function to do 2d convolution
    :param x: input 4D tensor.
    :param kernel_size: integer, the size of kernel, e.g. 3x3, 5x5.
    :param num_outputs: integer, the number of output filters.
    :param depth_outputs:.
    :param stride: a list of length 3 `[in_depth, in_height, in_width]`.
        Can be an int if both strides are the same. Note that presently
        both strides must have the same value.
    :param padding: one of `"VALID"` or `"SAME"`.
    :param reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    :param padding: one of `"VALID"` or `"SAME"`.
    :param trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    :param name: Optional scope for `variable_scope`.
    :return:
    """
    with tf.variable_scope(name, default_name='conv3d', reuse=reuse):
        in_channels = x.get_shape().dims[4].value
        filters = create_variables('weights',
                                   [depth_outputs, kernel_size, kernel_size, in_channels, num_outputs],
                                   weight_decay=weight_decay)
        out = tf.nn.conv3d(x,
                           filters,
                           strides=[1] + stride + [1],
                           padding=padding,
                           name=name)
        out = slim.bias_add(out, reuse=reuse, trainable=trainable, scope=name)
        return out


def deconv2d(x,
             kernel_size,
             num_outputs,
             filters=None,
             factor=2,
             padding='SAME',
             weight_decay=1e-4,
             reuse=False,
             trainable=True,
             addbias=False,
             name=None):
    """
    Helper function to do 2d convolution
    :param x: input 4D tensor
    :param kernel_size: integer, the size of kernel, e.g. 3x3, 5x5
    :param num_outputs: integer, the number of output filters.
    :param filters: input 4D tensor of filter. If filter is None,
      a variable created as [kernel_size, kernel_size, in_channels, out_channels].
    :param factor: scale of up-sampling.
    :param padding: one of `"VALID"` or `"SAME"`.
    :param reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    :param padding: one of `"VALID"` or `"SAME"`.
    :param trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    :param addbias: this parameter is valid for filter is not None.
    :param name: Optional scope for `variable_scope`.
    :return:
    """
    if filters is None:
        return slim.conv2d_transpose(x,
                                     num_outputs,
                                     kernel_size,
                                     stride=factor,
                                     padding=padding,
                                     activation_fn=None,
                                     weights_regularizer=tf.contrib.layers.l2_regularizer(scale=weight_decay),
                                     reuse=reuse,
                                     trainable=trainable,
                                     scope=name)
    else:
        batch = x.get_shape().dims[0].value
        height = x.get_shape().dims[1].value * factor
        width = x.get_shape().dims[2].value * factor

        out_channel = filters.get_shape().dims[2].value

        out = tf.nn.conv2d_transpose(x,
                                     filters,
                                     output_shape=[batch, height, width, out_channel],
                                     strides=[1, factor, factor, 1],
                                     padding=padding,
                                     name=name)
        if addbias:
            out = slim.bias_add(out, reuse=reuse, trainable=trainable, scope=name)
        return out


def deconv3d(x,
             kernel_size,
             num_outputs,
             depth_outputs,
             factor=2,
             padding='SAME',
             weight_decay=1e-4,
             reuse=False,
             trainable=True,
             name=None):
    """
    Helper function to do 2d convolution
    :param x: input 4D tensor.
    :param kernel_size: integer, the size of kernel, e.g. 3x3, 5x5.
    :param num_outputs: integer, the number of output filters.
    :param depth_outputs:.
    :param factor: scale of up-sampling.
    :param stride: a list of length 3 `[in_depth, in_height, in_width]`.
        Can be an int if both strides are the same. Note that presently
        both strides must have the same value.
    :param padding: one of `"VALID"` or `"SAME"`.
    :param reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    :param padding: one of `"VALID"` or `"SAME"`.
    :param trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    :param name: Optional scope for `variable_scope`.
    :return:
    """
    with tf.variable_scope(name, default_name='conv3d', reuse=reuse):
        batch = x.get_shape().dims[0].value
        in_depth = x.get_shape().dims[1].value * factor
        height = x.get_shape().dims[2].value * factor
        width = x.get_shape().dims[3].value * factor
        in_channels = x.get_shape().dims[4].value

        filters = create_variables('weights',
                                   [depth_outputs, kernel_size, kernel_size, num_outputs, in_channels],
                                   weight_decay=weight_decay)
        out = tf.nn.conv3d(x,
                           filters,
                           output_shape=[batch, depth_outputs, height, width, num_outputs],
                           strides=[1, int(depth_outputs/in_depth), factor, factor, 1],
                           padding=padding,
                           name=name)
        out = slim.bias_add(out, reuse=reuse, trainable=trainable, scope=name)
        return out


def dynamic_filter(x, filters, filter_size, name=None):
    with tf.name_scope(name, default_name="xcross_conv"):
        patches = tf.extract_image_patches(x,
                                           [1, filter_size, filter_size, 1],
                                           [1, 1, 1, 1],
                                           [1, 1, 1, 1],
                                           padding='SAME')
        out = tf.reduce_sum(tf.multiply(patches, filters), axis=3, keep_dims=True)
    return out


def relu(x, name=None):
    return tf.nn.relu(features=x)


def lrelu(x, alpha=0.2, name=None):
    return tf.maximum(x, alpha*x)


def upsampling(x, factor, out_channels, conv_type, name=None):
    with tf.variable_scope(name, default_name='deconv'):
        batch_n = x.get_shape().dims[0].value
        new_height = x.get_shape().dims[1].value * factor
        new_width = x.get_shape().dims[2].value * factor
        in_channels = x.get_shape().dims[3]
        output_shape = [batch_n, new_height, new_width, out_channels]
        kernel = get_bilinear_filter2('weights', factor, in_channels, out_channels)
        biases = create_variables(name='biases',
                                  shape=out_channels,
                                  initializer=tf.constant_initializer(0.0))
        out = tf.nn.conv2d_transpose(value=x, filter=kernel,
                                     output_shape=output_shape,
                                     strides=[1, factor, factor, 1],
                                     padding=conv_type)
        out = tf.nn.bias_add(value=out, bias=biases)
    return out


def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def get_bilinear_filter1(name, factor, in_channels, out_channels):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """

    filter_size = get_kernel_size(factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        in_channels,
                        out_channels), dtype=np.float32)

    bilinear = upsample_filt(filter_size)

    for i in range(out_channels):
            weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return create_variables(name, shape=weights.shape, initializer=init, wd=1e-4)


def get_bilinear_filter2(name, factor, in_channels, out_channels):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    filter_size = get_kernel_size(factor)

    f = np.ceil(filter_size/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)

    weights = np.zeros((filter_size,
                        filter_size,
                        out_channels,
                        in_channels), dtype=np.float32)

    bilinear = np.zeros([filter_size, filter_size])
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))

    for i in range(out_channels):
            weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return create_variables(name, shape=weights.shape, initializer=init)


def phase_shift(im, r):
    _, a, b, c = im.get_shape().as_list()
    bsize = tf.shape(im)[0]  # Handling Dimension(None) type for undefined batch dim
    X = tf.reshape(im, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))
    X = tf.split(1, a, X)
    X = tf.concat(3, [x for x in X])
    X = tf.split(2, b, X)
    X = tf.concat(4, [x for x in X])
    return tf.reshape(X, (bsize, a * r, b * r, 1))


def upsubpix(x, r, color=False):
        if color:
            x = tf.concat(3, [phase_shift(c, r) for c in tf.split(3, 3, x)])
        else:
            x = phase_shift(x, r)
        return x


def seq_bilinear_up_layer(value, factor, reuse, name='seq_bilinear'):
    shape = value.get_shape().as_list()
    if len(shape) == 5:
        value = tf.transpose(value, [1, 0, 2, 3, 4])
        value = tf.reshape(value, [-1] + shape[2:])
        value = tf.split(0, shape[1], value)
    if not _is_sequence(value):
        raise ValueError('`value` must be a list')
    batch_n = value[0].get_shape().as_list()[0]
    new_height = value[0].get_shape().as_list()[1] * factor
    new_width = value[0].get_shape().as_list()[2] * factor
    channels = value[0].get_shape().as_list()[3]
    with tf.variable_scope(name, reuse=reuse):
        kernel = get_bilinear_filter2('weights', factor, channels, channels)
        out = [tf.nn.conv2d_transpose(value=x, filter=kernel,
                                      output_shape=[batch_n, new_height, new_width, channels],
                                      strides=[1, factor, factor, 1]) for x in value]
    if len(shape) == 5:
        out = tf.pack(out)
        out = tf.transpose(out, [1, 0, 2, 3, 4])
    return out


def conv_rnn_layer(x, filter_size, in_channels, out_channels, conv_type, activation, reuse, name=None):
    def _step1((t_, _), x_):
        with tf.variable_scope('rnn_block_1', reuse=reuse):
            v = convolution_layer(x_, filter_size, in_channels, out_channels,
                                  conv_type, reuse, name='conv_v')
            t = convolution_layer(t_, filter_size, in_channels, out_channels,
                                  conv_type, reuse, name='conv_t')
            h = v + t
        return x_, h

    def _step2(r_, h):
        with tf.variable_scope('rnn_block_2', reuse=reuse):
            o = convolution_layer(r_, filter_size, out_channels, out_channels,
                                  conv_type, reuse, name='conv_r')
            if activation is not None:
                h = activation(tf.add(o, h))
        return h

    output_shape = x.get_shape().as_list()[1:-1] + [out_channels]

    with tf.variable_scope(name, reuse=reuse):
        _, out = tf.scan(fn=_step1, elems=x, initializer=(x[0, :], tf.zeros(output_shape)), name=name+'_step_1')
        out = tf.scan(fn=_step2, elems=out, initializer=out[0, :], name=name+'_step_2')

    return out


def residual(x, filter_size, in_channels, out_channels, conv_type, reuse, name=None):
    with tf.variable_scope(name, reuse=reuse):
        out = convolution_layer(x, filter_size, in_channels, out_channels, conv_type, reuse, 2*int(name))
        out = relu_layer(out, reuse, 2*int(name))
        out = convolution_layer(out, filter_size, in_channels, out_channels, conv_type, reuse, 2*int(name)+1)
        out = tf.add(x, out, name='identity')
        out = relu_layer(out, reuse, 2*int(name)+1)
        return out
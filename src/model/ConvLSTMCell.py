import numpy as np
from tensorflow.python.ops.array_ops import concat, reshape, split
from tensorflow.python.ops.init_ops import zeros_initializer, constant_initializer
from tensorflow.python.ops.math_ops import sigmoid, tanh
from tensorflow.python.ops.gen_nn_ops import conv2d as convolution
from tensorflow.python.ops.rnn_cell import LSTMStateTuple, RNNCell
from tensorflow.python.ops.variable_scope import get_variable, variable_scope
import logging
import tensorflow as tf

class ConvLSTMCell(RNNCell):
    """A LSTM cell with convolutions instead of multiplications.

    Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
    """

    def __init__(self, filter_size, num_units, height, width, input_size=None, state_is_tuple=True,
                 forget_bias=1.0, activation=tf.tanh, reuse=False):
        """Initialize the basic LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        if not state_is_tuple:
          logging.warn("%s: Using a concatenated state is slower and will soon be "
                       "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
          logging.warn("%s: The input_size parameter is deprecated.", self)

        self._height = height
        self._width = width
        self._filters = num_units
        self._kernel = [filter_size, filter_size]
        self._initializer = self.orthogonal_initializer()
        self._forget_bias = forget_bias
        self._activation = activation
        self._resue = reuse

    @property
    def state_size(self):
        size = self._height * self._width * self._filters
        return LSTMStateTuple(size, size)

    @property
    def output_size(self):
        return self._height * self._width * self._filters

    def _orthogonal(self, shape):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return q.reshape(shape)

    def _initializer(self, shape, dtype=tf.float32, partition_info=None):
      return tf.constant(self._orthogonal(shape), dtype)

    def orthogonal_initializer(self):
        return self._initializer

    def __call__(self, input, state, scope=None):
        """Convolutional Long short-term memory cell (ConvLSTM)."""
        with variable_scope(scope or type(self).__name__):
            previous_memory, previous_output = state

        with variable_scope('Expand'):
            samples = input.get_shape()[0].value
            shape = [samples, self._height, self._width]
            input = reshape(input, shape + [-1])
            previous_memory = reshape(previous_memory, shape + [self._filters])
            previous_output = reshape(previous_output, shape + [self._filters])

        with variable_scope('Convolve'):
            channels = input.get_shape()[-1].value
            filters = self._filters
            gates = 4 * filters if filters > 1 else 4

        with variable_scope('Input'):
            x = input
            n = channels
            m = gates
            W = get_variable('Weights', self._kernel + [n, m], initializer=self._initializer)
            Wxh = convolution(x, W, [1, 1, 1, 1], 'SAME')

        with variable_scope('Hidden'):
            x = previous_output
            n = filters
            m = gates
            W = get_variable('Weights', self._kernel + [n, m], initializer=self._initializer)
            Whh = convolution(x, W, [1, 1, 1, 1], 'SAME')
            y = Wxh + Whh
            y += get_variable('Biases', [m], initializer=zeros_initializer)

        input, input_gate, forget_gate, output_gate = split(3, 4, y)  # TODO Update to TensorFlow 1.0.

        with variable_scope('LSTM'):
            memory = (previous_memory * sigmoid(forget_gate + self._forget_bias) + sigmoid(input_gate) * self._activation(input))
            output = self._activation(memory) * sigmoid(output_gate)

        with variable_scope('Flatten'):
            shape = [-1, self._height * self._width * self._filters]
            output = reshape(output, shape)
            memory = reshape(memory, shape)

        return output, LSTMStateTuple(memory, output)


class ConvRNNCell(RNNCell):
    """A RNN cell with convolutions instead of multiplications.
    """

    def __init__(self, filter_size, num_units, height, width, input_size=None, activation=tanh, reuse=False):
        """Initialize the basic RNN cell.

        Args:
          num_units: int, The number of units in the RNN cell.
          input_size: Deprecated and unused.
          activation: Activation function of the inner states.
        """
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)

        self._height = height
        self._width = width
        self._filters = num_units
        self._kernel = [filter_size, filter_size]
        self._initializer = self.orthogonal_initializer()
        self._activation = activation
        self._resue = reuse

    @property
    def state_size(self):
        return self._height * self._width * self._filters

    @property
    def output_size(self):
        return self._height * self._width * self._filters

    def _orthogonal(self, shape):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return q.reshape(shape)

    def _initializer(self, shape, dtype=tf.float32, partition_info=None):
        return tf.constant(self._orthogonal(shape), dtype)

    def orthogonal_initializer(self):
        return self._initializer

    def __call__(self, input, state, scope=None):
        with variable_scope(scope or type(self).__name__):
            samples = input.get_shape()[0].value
            shape = [samples, self._height, self._width]
            input = reshape(input, shape + [-1])
            state = reshape(state, shape + [self._filters])

            channels = input.get_shape()[-1].value
            filters = self._filters
            Wx = get_variable('Weights_input', self._kernel + [channels, filters], initializer=self._initializer)
            Wh = get_variable('Weights_hidden', self._kernel + [filters, filters], initializer=self._initializer)
            Wxh = convolution(input, Wx, [1, 1, 1, 1], 'SAME')
            Whh = convolution(state, Wh, [1, 1, 1, 1], 'SAME')
            y = Wxh + Whh
            y += get_variable('Biases', [filters], initializer=zeros_initializer)
            if self._activation:
                output = self._activation(y)
            else:
                output = y

            shape = [-1, self._height * self._width * self._filters]
            output = reshape(output, shape)

        return output, output


def flatten(tensor):
    samples, timesteps, height, width, filters = tensor.get_shape().as_list()
    return reshape(tensor, [samples, timesteps, height * width * filters])


def expand(tensor, height, width):
    samples, timesteps, features = tensor.get_shape().as_list()
    return reshape(tensor, [samples, timesteps, height, width, -1])

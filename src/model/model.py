import os
import sys
import abc
import collections
from tensorflow.python.ops import array_ops, control_flow_ops
from ops import *
from src.utility.helper import *
from ConvLSTMCell import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ModelParams = collections.namedtuple('ModelParams', [
    'kernel_number', 'kernel_size', 'weight_decay', 'padding_type'
])


def variables_to_restore(scope=None, strip_scope=False):
    """Returns a list of variables to restore for the specified list of methods.
    It is supposed that variable name starts with the method's scope (a prefix
    returned by _method_scope function).
    Args:
        methods_names: a list of names of configurable methods.
        strip_scope: if True will return variable names without method's scope.
        If methods_names is None will return names unchanged.
        model_scope: a scope for a whole model.
    Returns:
        a dictionary mapping variable names to variables for restore.
    """
    if scope:
        variable_map = {}
        method_variables = slim.get_variables_to_restore(include=[scope])
        for var in method_variables:
            if strip_scope:
                var_name = var.op.name[len(scope) + 1:]
            else:
                var_name = var.op.name
            variable_map[var_name] = var
        return variable_map
    else:
        return {v.op.name: v for v in slim.get_variables_to_restore()}


class Model(object):
    """Class to create the Super Resolution Model."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, is_training=True, is_reuse=False, **kwargs):
        """Initialized model parameters."""
        super(Model, self).__init__()
        self.layers = dict()
        params = {'is_training': is_training, 'is_reuse': is_reuse}
        params.update(kwargs)
        MethodParams = collections.namedtuple('MethodParams', params.keys())
        self._params = MethodParams(**params)
        self._mparams = ModelParams(kernel_number=32,
                                    kernel_size=3,
                                    weight_decay=1e-4,
                                    padding_type='SAME')

    @abc.abstractmethod
    def inference(self, input, **kwargs):
        pass

    @abc.abstractmethod
    def create_loss(self, prediction, label, **kwargs):
        pass

    @abc.abstractmethod
    def create_summaries(self, summwriter):
        pass

    def vdsr(self, x):
        flags = self._flags
        layer_list = self._layer_list
        reuse = self._reuse
        ct = flags.conv_type
        fz = flags.filter_size
        wd = flags.weight_decay
        fn = flags.filter_number

        layer_list.append(x)

        out = self._convolution_layer(x, fz, 1, fn, wd, ct, reuse, 1)
        layer_list.append(out)

        out = self._relu_layer(out, reuse, 1)
        layer_list.append(out)

        for i in range(2, 10):
            out = self._convolution_layer(out, fz, fn, fn, wd, ct, reuse, i)
            layer_list.append(out)

            out = self._relu_layer(out, reuse, i)
            layer_list.append(out)

        out = self._convolution_layer(out, fz, fn, 1, wd, ct, reuse, 10)
        out = tf.add(out, x, name='identity')
        layer_list.append(out)

        return out

    def vdsr_add_bn(self, x):
        flags = self._flags
        layer_list = self._layer_list
        reuse = self._reuse
        ct = flags.conv_type
        fz = flags.filter_size
        wd = flags.weight_decay
        fn = flags.filter_number

        out = self._convolution_layer(x, fz, 1, fn, wd, ct, reuse, 'conv1')
        out = self._batch_normalization_layer(out, name='bn1')
        out = self._relu_layer(out, reuse, 'relu1')

        for i in range(2, 10):
            out = self._convolution_layer(out, fz, fn, fn, wd, ct, reuse, 'conv%d' % i)
            out = self._batch_normalization_layer(out, 'bn%d' % i)
            out = self._relu_layer(out, reuse, 'relu%d' % i)

        out = self._convolution_layer(out, fz, fn, 1, wd, ct, reuse, 'conv10')
        out = tf.add(out, x, name='identity')
        layer_list.append(out)

        return out

    def espcn(self, x):
        flags = self._flags
        layer_list = self._layer_list
        reuse = self._reuse
        us = self._upscale
        ct = flags.conv_type
        fz = flags.filter_size
        wd = flags.weight_decay
        fn = flags.filter_number

        out = self._convolution_layer(x, fz, 1, fn, wd, ct, reuse, 'conv1')
        layer_list.append(out)

        out = self._relu_layer(out, reuse, 'relu1')

        for i in range(2, 10):
            out = self._convolution_layer(out, fz, fn, fn, wd, ct, reuse, 'conv%d' % i)
            if i == 9:
                out = tf.add(out, layer_list[0], name='identity')
            out = self._relu_layer(out, reuse, 'relu%d' % i)

        out = self._convolution_layer(out, fz, fn, us**2, wd, ct, reuse, 'conv10')

        out = self._subpix_deconv2d(out, us)

        return out

    def deconv(self, x):
        flags = self._flags
        layer_list = self._layer_list
        reuse = self._reuse
        us = self._upscale
        ct = flags.conv_type
        fz = flags.filter_size
        wd = flags.weight_decay
        fn = flags.filter_number
        num = np.log2(us)

        layer_list.append(x)

        out = self._convolution_layer(x, fz, 1, fn, wd, ct, reuse, 1)
        layer_list.append(out)

        out = self._relu_layer(out, reuse, 1)
        layer_list.append(out)

        for i in range(2, 8):
            out = self._convolution_layer(out, fz, fn, fn, wd, ct, reuse, i)
            if i == 7:
                out = tf.add(out, layer_list[1], name='identity')
            layer_list.append(out)

            out = self._relu_layer(out, reuse, i)
            layer_list.append(out)

        for i in range(num):
            shape = out.get_shape().as_list()
            out = self._deconvolution_layer(out, fz, fn, fn, wd, ct,
                                            (shape[0], 2*shape[1], 2*shape[2], fn), (1, 2, 2, 1), reuse, 8+i)
            layer_list.append(out)
            out = self._relu_layer(out, reuse, 8+i)
            layer_list.append(out)

        out = self._convolution_layer(out, fz, fn, 1, wd, ct, reuse, 10)
        layer_list.append(out)

        return out

    def bilinear_conv(self, x):
        flags = self._flags
        layer_list = self._layer_list
        reuse = self._reuse
        us = self._upscale
        ct = flags.conv_type
        fz = flags.filter_size
        wd = flags.weight_decay
        fn = flags.filter_number
        num = int(np.log2(us))

        out = self._convolution_layer(x, fz, 1, fn, wd, ct, reuse, 'conv1')
        layer_list.append(out)
        out = self._relu_layer(out, reuse, 'relu1')

        for i in range(2, 8):
            out = self._convolution_layer(out, fz, fn, fn, wd, ct, reuse, 'conv%d' % i)
            if i == 7:
                out = tf.add(out, layer_list[0], name='identity')
            out = self._relu_layer(out, reuse, 'conv%d' % i)

        for i in range(num):
            out = bilinear_layer(out, 2, reuse, 'bilinear%d' % (i+1))
            out = self._convolution_layer(out, fz, fn, fn, wd, ct, reuse, 'conv%d' % (i+8))
            out = self._relu_layer(out, reuse, 'relu%d' % (i+8))

        out = self._convolution_layer(out, fz, fn, 1, wd, ct, reuse, 'conv10')

        return out

    def sys_skipping(self, x):
        flags = self._flags
        layer_list = self._layer_list
        reuse = self._reuse
        ct = flags.conv_type
        fz = flags.filter_size
        wd = flags.weight_decay
        fn = flags.filter_number

        layer_list.append(x)

        out = self._convolution_layer(x, fz, 1, fn, wd, ct, reuse, 1)
        layer_list.append(out)

        out = self._relu_layer(out, reuse, 1)
        layer_list.append(out)

        for i in range(2, 10):
            out = self._convolution_layer(x, fz, 1, fn, wd, ct, reuse, i)
            if i == 6:
                out = tf.add(out, layer_list[7], name='identity')
            if i == 7:
                out = tf.add(out, layer_list[5], name='identity')
            if i == 8:
                out = tf.add(out, layer_list[3], name='identity')
            if i == 9:
                out = tf.add(out, layer_list[1], name='identity')
            out = self._relu_layer(out, reuse, i)
            layer_list.append(out)

        out = self._convolution_layer(out, fz, fn, 1, wd, ct, reuse, 10)
        out = tf.add(out, x, name='identity')
        layer_list.append(out)

        return out

    def est(self, x):
        flags = self._flags
        reuse = self._reuse
        ct = flags.conv_type
        fz = flags.filter_size
        wd = flags.weight_decay
        fn = flags.filter_number

        out = self._convolution_layer(x, fz, 1, fn, wd, ct, reuse, 1)
        out = self._relu_layer(out, reuse, 1)

        for i in range(4):
            out = self._residual(out, fz, fn, fn, wd, ct, reuse, i+1)

        out = self._convolution_layer(out, fz, fn, 1, wd, ct, reuse, 10)
        out = tf.add(out, x, name='identity')

        return out

    def bidirection_rnn(self, x):
        flags = self._flags
        reuse = self._reuse
        us = self._upscale
        ct = flags.conv_type
        fz = flags.filter_size
        wd = flags.weight_decay
        fn = flags.filter_number
        num = int(np.log2(us))
        layer_list = self._layer_list
        shape = x.get_shape().as_list()
        batch_size = shape[0]
        timesteps = shape[1]
        height = shape[2]
        width = shape[3]
        channel = shape[4]

        # with tf.variable_scope('birnn', reuse=self._reuse):
        #     cell_fw = ConvRNNCell(fz, fn, height, width, reuse=self._reuse, activation=None)
        #     cell_bw = ConvRNNCell(fz, fn, height, width, reuse=self._reuse, activation=None)
        #     state_fw = cell_fw.zero_state(batch_size, tf.float32)
        #     state_bw = cell_bw.zero_state(batch_size, tf.float32)
        #     outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=flatten(x),
        #                                                  sequence_length=[10] * batch_size,
        #                                                  initial_state_fw=state_fw,
        #                                                  initial_state_bw=state_bw)
        #     out = expand(outputs[0], height, width) + expand(outputs[1], height, width)

        with tf.variable_scope('birnn', reuse=self._reuse):
            x = array_ops.transpose(x, perm=(1, 0, 2, 3, 4), name='time-major')  # time-major
            x_rev = array_ops.reverse_v2(x, axis=[0]) # reverse for bi-direction
            out_fw = self._conv_rnn_layer(x, fz, 1, fn, 0.0, ct, tf.nn.relu, reuse, name='fw')
            out_bw = self._conv_rnn_layer(x_rev, fz, 1, fn, 0.0, ct, tf.nn.relu, reuse, name='bw')
            out_bw = array_ops.reverse_v2(out_bw, axis=[0])
        out = tf.add(out_fw, out_bw)
        out = array_ops.transpose(out, perm=(1, 0, 2, 3, 4), name='batch-major')

        out = tf.reshape(out, [batch_size * timesteps, height, width, fn])
        layer_list.append(out)
        out = self._relu_layer(out, reuse, 'relu1')

        out = self._convolution_layer(out, fz, fn, fn, wd, ct, reuse, 'conv2')
        layer_list.append(out)
        out = self._relu_layer(out, reuse, 'relu2')

        out = self._convolution_layer(out, fz, fn, fn, wd, ct, reuse, 'conv3')
        layer_list.append(out)
        out = self._relu_layer(out, reuse, 'relu3')

        out = self._convolution_layer(out, fz, fn, fn, wd, ct, reuse, 'conv4')
        out = self._relu_layer(out, reuse, 'relu4')

        out = self._convolution_layer(out, fz, fn, fn, wd, ct, reuse, 'conv5')
        out = tf.add(out, layer_list[2], name='identity1')
        out = self._relu_layer(out, reuse, 'relu5')

        out = self._convolution_layer(out, fz, fn, fn, wd, ct, reuse, 'conv6')
        out = tf.add(out, layer_list[1], name='identity2')
        out = self._relu_layer(out, reuse, 'relu6')

        out = self._convolution_layer(out, fz, fn, fn, wd, ct, reuse, 'conv7')
        out = tf.add(out, layer_list[0], name='identity3')
        out = self._relu_layer(out, reuse, 'conv7')

        for i in range(num):
            # out = bilinear_layer(out, 2, reuse, 'bilinear%d' % (i + 1))
            out = self._convolution_layer(out, fz, fn, fn, wd, ct, reuse, 'conv%d' % (i + 8))
            out = self._relu_layer(out, reuse, 'relu%d' % (i + 8))

        out = self._convolution_layer(out, fz, fn, 1, wd, ct, reuse, 'conv10')
        out = tf.reshape(out, [batch_size, timesteps, height * us, width * us, channel])

        return out


class OpticalFlowWithAssemble(Model):
    """One of MODEL which uses ASSEMBLE mechanism combined with optical-flow dynamic filter."""
    def __init__(self, *args, **kwargs):
        super(OpticalFlowWithAssemble, self).__init__(*args, **kwargs)

    def optical_flow_dynamic_filter(self, x_slice, flow_slice):
        num_outputs = self._mparams.kernel_number
        kernel_size = self._mparams.kernel_size

        with tf.name_scope('filter_generating_net'):

            assert len(x_slice) == len(flow_slice), 'The optical-flow input should equal the data.'

            with tf.variable_scope('flow_filter_conv'):
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu):
                    op_filter = [slim.conv2d(i, num_outputs, [kernel_size, kernel_size]) for i in flow_slice]
                    op_filter = [slim.conv2d(i, kernel_size ** 2, [kernel_size, kernel_size]) for i in op_filter]
                    op_filter = [tf.nn.softmax(i) for i in op_filter]

        with tf.name_scope('dynamic_filtering_layer'):
            x_warp = [dynamic_filter(image, flow, kernel_size) for image, flow in zip(x_slice, op_filter)]

        return x_warp, op_filter

    def inference(self, inputs, **kwargs):
        x, flow = inputs
        num_outputs = self._mparams.kernel_number
        kernel_size = self._mparams.kernel_size
        padding_type = self._mparams.padding_type

        self.layers['input'] = x
        self.layers['flow'] = flow

        with tf.variable_scope('optical_assemble_v1', reuse=self._params.is_reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=None,
                                padding=padding_type,
                                weights_regularizer=slim.l2_regularizer(self._mparams.weight_decay)):

                x_warp, flow_filter = self.optical_flow_dynamic_filter(x, flow)
                self.layers['flow_filter'] = flow_filter
                self.layers['input_warped'] = x_warp

                conv_0 = tf.concat(x_warp, 3)
                self.layers['conv0'] = conv_0

                conv_1 = slim.conv2d(conv_0, num_outputs, kernel_size)
                self.layers['conv1'] = conv_1
                conv_1 = relu(conv_1)

                conv_2 = slim.conv2d(conv_1, num_outputs, kernel_size)
                self.layers['conv2'] = conv_2
                conv_2 = relu(conv_2)

                conv_3 = slim.conv2d(conv_2, num_outputs, kernel_size)
                self.layers['conv3'] = conv_3
                conv_3 = relu(conv_3)

                conv_4 = slim.conv2d(conv_3, num_outputs, kernel_size)
                self.layers['conv4'] = conv_4
                conv_4 = relu(conv_4)

                conv_5 = slim.conv2d(conv_4, num_outputs, kernel_size) + self.layers['conv3']
                self.layers['conv5'] = conv_5
                conv_5 = relu(conv_5)

                conv_6 = slim.conv2d(conv_5, num_outputs, kernel_size) + self.layers['conv2']
                self.layers['conv6'] = conv_6
                conv_6 = relu(conv_6)

                conv_7 = slim.conv2d(conv_6, num_outputs, kernel_size) + self.layers['conv1']
                self.layers['conv7'] = conv_7
                conv_7 = relu(conv_7)

                conv_up_1 = upsampling(conv_7, 2, num_outputs, padding_type)
                self.layers['conv_up1'] = conv_up_1
                conv_up_1 = relu(conv_up_1)

                conv_up_2 = upsampling(conv_up_1, 2, num_outputs, padding_type)
                self.layers['conv_up2'] = conv_up_2
                conv_up_2 = relu(conv_up_2)

                conv_8 = slim.conv2d(conv_up_2, 1, kernel_size)
                self.layers['conv8'] = conv_8

                out = relu(conv_8)
                self.layers['output'] = out

                return out

    def create_loss(self, prediction, label, **kwargs):
        tf.losses.mean_squared_error(prediction, label)
        total_loss = tf.losses.get_total_loss(add_regularization_losses=self._params.is_training)
        return total_loss

    def create_summaries(self, summwriter):
        """Creates all summaries for the model.
        Args:
            summwriter: SummaryToWrite namedtuple.
        Returns:
            A list of evaluation ops
        """
        def sname(label):
            prefix = 'train' if self._params.is_training else 'eval'
            return '%s/%s' % (prefix, label)

        def use_metric(name, value_update_tuple):
            names_to_values[name] = value_update_tuple[0]
            names_to_updates[name] = value_update_tuple[1]

        max_outputs = 5
        names_to_values = {}
        names_to_updates = {}

        for i, slip in enumerate(summwriter.input.images):
            tf.summary.image('input_%d' % i, slip, max_outputs=max_outputs)

        for i, slip in enumerate(summwriter.model['input_warped']):
            tf.summary.image('warped_%d' % i, slip, max_outputs=max_outputs)

        tf.summary.image('predication', summwriter.output, max_outputs=max_outputs)
        tf.summary.image('label', summwriter.label, max_outputs=max_outputs)

        flow = summwriter.model['flow'][2]
        flow = reshape(flow, [-1, 2])

        flow_filter = summwriter.model['flow_filter'][2]
        flow_filter = tf.reshape(flow_filter, [-1, 3, 3])

        if self._params.is_training:
            tf.summary.scalar('Loss', summwriter.loss)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            return None
        else:
            use_metric('Loss', slim.metrics.streaming_mean(summwriter.loss))
            use_metric('PSNR', slim.metrics.streaming_mean(pnsr_tf(summwriter.output, summwriter.label)))

            mask = tf.reduce_any(tf.greater(tf.abs(flow), 1), axis=1)
            flow = tf.as_string(tf.boolean_mask(flow, mask))
            flow_filter = tf.as_string(tf.boolean_mask(flow_filter, mask))
            for i in range(5):
                tf.summary.text('flow', flow[i])
                tf.summary.text('flow_filter', flow_filter[i])



def create_init_fn_to_restore(master_checkpoint):
    """Creates an init operations to restore weights from various checkpoints.
    Args:
        master_checkpoint: path to a checkpoint which contains all weights for the whole model.
    Returns:
        a function to run initialization ops.
    """

    all_assign_ops = []
    all_feed_dict = {}

    def assign_from_checkpoint(variables, checkpoint):
        logging.info('Request to re-store %d weights from %s', len(variables), checkpoint)
        if not variables:
            logging.error('Can\'t find any variables to restore.')
            sys.exit(1)
        assign_op, feed_dict = slim.assign_from_checkpoint(checkpoint, variables)
        all_assign_ops.append(assign_op)
        all_feed_dict.update(feed_dict)

    if master_checkpoint:
        assign_from_checkpoint(variables_to_restore(), master_checkpoint)

    def init_assign_fn(sess):
        logging.info('Restoring checkpoint(s)')
        sess.run(all_assign_ops, all_feed_dict)

    return init_assign_fn














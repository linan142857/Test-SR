# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from tensorflow import app
from tensorflow.contrib.tfprof import model_analyzer
from src.utility import common_flags
from src.utility.data_provider import *
from src.model.model import *
from src.utility.common_flags import dops, SummaryToWrite

FLAGS = flags.FLAGS
common_flags.define()

TrainingHParams = collections.namedtuple('TrainingHParams', [
    'learning_rate',
    'optimizer',
    'momentum',
    'decay_steps',
    'decay_rate'
])

hparams = TrainingHParams(
    learning_rate=FLAGS.learning_rate,
    optimizer=FLAGS.optimizer,
    momentum=FLAGS.momentum,
    decay_steps=FLAGS.decay_steps,
    decay_rate=FLAGS.decay_rate)


def prepare_training_dir():
    if not tf.gfile.Exists(FLAGS.train_log_dir):
        logging.info('Create a new training directory %s', FLAGS.train_log_dir)
        tf.gfile.MakeDirs(FLAGS.train_log_dir)
    else:
        if FLAGS.reset_train_dir:
            logging.info('Reset the training directory %s', FLAGS.train_log_dir)
            tf.gfile.DeleteRecursively(FLAGS.train_log_dir)
            tf.gfile.MakeDirs(FLAGS.train_log_dir)
        else:
            logging.info('Use already existing training directory %s', FLAGS.train_log_dir)


def calculate_graph_metrics():
    param_stats = model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    return param_stats.total_parameters


def create_optimizer(hparams):
    """Creates optimized based on the specified flags."""
    learning_rate = tf.train.exponential_decay(hparams.learning_rate,
                                               slim.get_or_create_global_step(),
                                               hparams.decay_steps,
                                               hparams.decay_rate,
                                               staircase=True)

    if hparams.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=hparams.momentum)
    elif hparams.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif hparams.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif hparams.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif hparams.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=hparams.momentum)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    return optimizer


def train(loss, init_fn, hparams):
    """Wraps slim.learning.train to run a training loop.
    Args:
        loss: a loss tensor
        init_fn: A callable to be executed after all other initialization is done.
        hparams: a model hyper parameters
    """
    optimizer = create_optimizer(hparams)

    train_op = slim.learning.create_train_op(
        loss,
        optimizer,
        summarize_gradients=False,
        clip_gradient_norm=FLAGS.clip_gradient_norm)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    slim.learning.train(
        train_op=train_op,
        logdir=FLAGS.train_log_dir,
        log_every_n_steps=FLAGS.log_steps,
        graph=loss.graph,
        number_of_steps=FLAGS.max_number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        init_fn=init_fn,
        session_config=config)


def main(_):
    prepare_training_dir()
    provider = DataProviderAssembleOP(
        dops.crop_frm_no,
        FLAGS.batch_size,
        augment=dops.is_aug,
        shuffle=True)

    provider.set_crop_config(dops.crop_size, dops.crop_stride, dops.scale)

    model = OpticalFlowWithAssemble(is_training=True, is_reuse=False)

    dataset = create_dataset(split_name=FLAGS.split_name)
    input_points = provider.get_data(dataset)

    end_points = model.inference([input_points.images, input_points.flows])
    total_loss = model.create_loss(end_points, input_points.labels)
    summwriter = SummaryToWrite(input=input_points,
                                output=end_points,
                                model=model.layers,
                                label=input_points.labels,
                                loss=total_loss)
    model.create_summaries(summwriter)

    checkpoint = FLAGS.checkpoint or tf.train.latest_checkpoint(FLAGS.train_log_dir)
    init_fn = create_init_fn_to_restore(checkpoint)

    if FLAGS.show_graph_stats:
        logging.info('Total number of weights in the graph: %s', calculate_graph_metrics())
    train(total_loss, init_fn, hparams)


if __name__ == '__main__':
    app.run()

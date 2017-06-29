"""Define flags are common for both train.py and eval.py scripts."""
import os
import sys
import logging
import collections
from tensorflow.python.platform import flags

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FLAGS = flags.FLAGS

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stderr,
    format='%(levelname)s '
    '%(asctime)s.%(msecs)06d: '
    '%(filename)s: '
    '%(lineno)d '
    '%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

SummaryToWrite = collections.namedtuple('SummaryToWrite', [
    'input',
    'output',
    'label',
    'model',
    'loss'
])

DataOptions = collections.namedtuple('DataOptions', [
    'crop_size',
    'crop_frm_no',
    'crop_stride',
    'scale',
    'is_up',
    'is_aug'])

dops = DataOptions(
    crop_size=33,
    crop_frm_no=3,
    crop_stride=17,
    scale=4,
    is_up=False,
    is_aug=False)


def define():
  """Define common flags."""
  '''training setting'''
  flags.DEFINE_string('dataset_name', 'video_assemble',
                      '''Name of the dataset.''')

  flags.DEFINE_string('train_log_dir', 'log/train_assemble_x3',
                      '''Directory where to write event logs.''')

  flags.DEFINE_string('dataset_dir', None,
                      '''Dataset root folder.''')

  flags.DEFINE_boolean('show_graph_stats', True,
                       '''Output model size stats to stderr.''')

  flags.DEFINE_integer('log_steps', 1000,
                       '''Display interval''')

  flags.DEFINE_integer('save_summaries_secs', 300,
                       '''The frequency with which summaries are saved, in seconds.''')

  flags.DEFINE_integer('save_interval_secs', 300,
                       '''Frequency in seconds of saving the model.''')

  flags.DEFINE_string('split_name', 'train',
                      '''Dataset split name to run evaluation for: test,train.''')

  flags.DEFINE_integer('batch_size', 32,
                       '''Train batch size''')

  flags.DEFINE_integer('max_number_of_steps', int(1e8),
                       '''The maximum number of gradient steps.''')

  flags.DEFINE_integer('decay_steps', int(1e6),
                       '''At which step to decay the learning rate''')

  flags.DEFINE_float('decay_rate', 0.5,
                     '''How much to decay the learning rate each time''')

  flags.DEFINE_float('learning_rate', 0.00001,
                     '''Initial learning rate''')

  flags.DEFINE_string('optimizer', '''rmsprop''',
                      'the optimizer to use')

  flags.DEFINE_string('momentum', 0.9,
                      'momentum value for the momentum optimizer if used')

  flags.DEFINE_float('clip_gradient_norm', 10.0,
                     '''If greater than 0 then the gradients would be clipped by it.''')

  flags.DEFINE_string('checkpoint', None,
                      '''Checkpoint to recover inception weights from.''')

  flags.DEFINE_boolean('reset_train_dir', False,
                       '''If true will delete all files in the train_log_dir''')

  flags.DEFINE_string('master', '',
                       '''BNS name of Tensorflow master to use''')

  '''evaluation setting'''
  flags.DEFINE_string('split_val_name', 'val',
                      '''Dataset split name to run evaluation for: test,train.''')

  flags.DEFINE_integer('val_batch_size', 1,
                       '''Train batch size''')

  flags.DEFINE_integer('num_batches', 32,
                       'Number of batches to run eval for.')

  flags.DEFINE_string('eval_log_dir', 'log/eval_assemble_x3',
                      '''Directory where the evaluation results are saved to.''')

  flags.DEFINE_integer('eval_interval_secs', 300,
                       ''''Frequency in seconds to run evaluations.''')

  flags.DEFINE_integer('number_of_steps', None,
                       '''Number of times to run evaluation.''')


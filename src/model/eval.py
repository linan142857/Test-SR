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

"""Script to evaluate a trained Attention OCR model.

A simple usage example:
python eval.py
"""
from tensorflow import app
from src.utility import common_flags
from src.utility.data_provider import *
from src.model.model import *
from src.utility.common_flags import dops, SummaryToWrite

FLAGS = flags.FLAGS
common_flags.define()


def main(_):
    if not tf.gfile.Exists(FLAGS.eval_log_dir):
        tf.gfile.MakeDirs(FLAGS.eval_log_dir)

    provider = DataProviderAssembleOP(
        dops.crop_frm_no,
        FLAGS.val_batch_size,
        augment=dops.is_aug,
        shuffle=False)

    model = OpticalFlowWithAssemble(is_training=False)

    data = create_dataset(split_name=FLAGS.split_val_name)
    input_points = provider.get_data(data)
    end_points = model.inference([input_points.images, input_points.flows])
    total_loss = model.create_loss(end_points, input_points.labels)
    summwriter = SummaryToWrite(input=input_points,
                                output=end_points,
                                model=model.layers,
                                label=input_points.labels,
                                loss=total_loss)
    eval_ops = model.create_summaries(summwriter)

    slim.get_or_create_global_step()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.train_log_dir,
        logdir=FLAGS.eval_log_dir,
        eval_op=eval_ops,
        num_evals=FLAGS.num_batches,
        eval_interval_secs=FLAGS.eval_interval_secs,
        max_number_of_evaluations=FLAGS.number_of_steps,
        session_config=config)

if __name__ == '__main__':
    app.run()

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

"""
Functions to read, decode and pre-process input data for the Model.
"""

import tensorflow as tf
import collections
import abc
from tensorflow.contrib import slim
from tensorflow.python.platform import flags
import src.data

FLAGS = flags.FLAGS

# A namedtuple to define a configuration for shuffled batch fetching.
#   num_batching_threads: A number of parallel threads to fetch data.
#   queue_capacity: a max number of elements in the batch shuffling queue.
#   min_after_dequeue: a min number elements in the queue after a dequeue, used
#   to ensure a level of mixing of elements.

ShuffleBatchConfig = collections.namedtuple('ShuffleBatchConfig', [
    'num_batching_threads', 'queue_capacity', 'min_after_dequeue'
])

DEFAULT_SHUFFLE_CONFIG = ShuffleBatchConfig(num_batching_threads=8, queue_capacity=3000, min_after_dequeue=1000)

# Tuple to store input data endpoints for the Model.
# It has following fields (tensors):
#    images: input images,
#      shape [batch_size x H x W x assemble];
#    labels: ground truth label,
#      shape=[batch_size x H' x W' x assemble];
#    flows: optical flow as the same spatial size of images,
InputEndpoints = collections.namedtuple(
    'InputEndpoints', ['images', 'labels', 'flows'])


def create_dataset(split_name):
    ds_module = getattr(src.data, FLAGS.dataset_name)
    return ds_module.get_split(split_name, dataset_dir=FLAGS.dataset_dir)


class DataProviderBase(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, batch_size, shuffle, augment):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.shuffle_config = DEFAULT_SHUFFLE_CONFIG

    @abc.abstractmethod
    def get_data(self, dataset):
        pass

    @abc.abstractmethod
    def preprocess_image(self, data):
        pass

    @abc.abstractmethod
    def augment_image(self, data):
        pass


class DataProviderAssembleOP(DataProviderBase):

    def __init__(self, assemble,
                 batch_size=32,
                 shuffle=False,
                 augment=False):
        super(DataProviderAssembleOP, self).__init__(batch_size, shuffle, augment)

        self.assemble = assemble
        self.crop_config = None

    def get_data(self, dataset):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=self.shuffle,
            common_queue_capacity=3 * self.batch_size,
            common_queue_min=self.batch_size)

        flow_shape = dataset.shape[-1]
        image_orig, label_orig, flow_orig = provider.get(['image', 'label', 'flow'])
        flow_orig = tf.reshape(flow_orig, flow_shape)

        # To solve the difference of training and evaluation.
        image_orig = self.preprocess_image(image_orig)
        label_orig = self.preprocess_image(label_orig)

        flow_flatten = tf.reshape(flow_orig, flow_shape[:-2] + [-1])
        input = tf.concat([image_orig, flow_flatten], 2)
        label = label_orig

        enqueue_many = False
        if self.crop_config is not None:
            enqueue_many = True
            input = tf.expand_dims(input, axis=0)
            input = self.crop_patches(input, self.crop_config.patch_size_x, self.crop_config.stride_x)
            label = tf.expand_dims(label, axis=0)
            label = self.crop_patches(label, self.crop_config.patch_size_y, self.crop_config.stride_y)

        inputs, labels = (tf.train.shuffle_batch(
            [input, label],
            batch_size=self.batch_size,
            num_threads=self.shuffle_config.num_batching_threads,
            capacity=self.shuffle_config.queue_capacity,
            min_after_dequeue=self.shuffle_config.min_after_dequeue,
            allow_smaller_final_batch=False,
            enqueue_many=enqueue_many))

        images = tf.slice(inputs, [0, 0, 0, 0], [-1, -1, -1, self.assemble])
        images = tf.split(images, self.assemble, 3)
        flows = tf.slice(inputs, [0, 0, 0, self.assemble], [-1, -1, -1, -1])
        flows = tf.split(flows, self.assemble, 3)

        return InputEndpoints(images=images,
                              labels=labels,
                              flows=flows)

    def set_crop_config(self, patch_size, stride, up_scale=None):
        CroppedConfig = collections.namedtuple('CroppedConfig', [
            'patch_size_x', 'stride_x', 'patch_size_y', 'stride_y'
        ])

        if up_scale is None:
            self.crop_config = CroppedConfig(patch_size_x=patch_size,
                                             stride_x=stride,
                                             patch_size_y=patch_size,
                                             stride_y=stride)
        else:
            assert isinstance(up_scale, int) and up_scale > 0, 'up_scale must be positive int'
            self.crop_config = CroppedConfig(patch_size_x=patch_size,
                                             stride_x=stride,
                                             patch_size_y=patch_size * up_scale,
                                             stride_y=stride * up_scale)

    def crop_patches(self, image, patch_size, stride):
        with tf.name_scope('CropImagePatches'):
            depth = image.get_shape().dims[-1].value
            image_patches = tf.extract_image_patches(image,
                                                     ksizes=[1, patch_size, patch_size, 1],
                                                     strides=[1, stride, stride, 1],
                                                     rates=[1, 1, 1, 1],
                                                     padding='VALID')
            image_patches = tf.reshape(image_patches, [-1, patch_size, patch_size, depth])
        return image_patches

    def preprocess_image(self, data):
        with tf.variable_scope('PreprocessImage'):
            data = tf.image.convert_image_dtype(data, dtype=tf.float32)
        return data

    def augment_image(self, data):
        with tf.name_scope('AugmentImage_Flipping'):
            data = tf.image.random_flip_up_down(data)
            data = tf.image.random_flip_left_right(data)

        return data

#
# def rgb2y(image):
#     """
#     Convert RGB image into Y of (YCRCB) https://en.wikipedia.org/wiki/YCRCB
#     L = R * 299/1000 + G * 587/1000 + B * 114/1000
#     """
#     with tf.name_scope('rgb2y'):
#         rgb = tf.unstack(image, axis=-1)
#         y = tf.add(rgb[0] * 299/1000 + rgb[1] * 587/1000 + rgb[2] * 114/1000)
#     return y
#
#
# def preprocess_image(hr, lr, fl, augment=False, convert=False):
#     """Pre-process one image for training or evaluation.
#     Args:
#     image: a [H x W x 3] uint8 tensor.
#     augment: optional, if True do random image distortion.
#     convert: A bool determine the channel of image.
#
#     Returns:
#         A float32 tensor of shape [H x W x 3] with RGB values in the required range.
#     """
#     with tf.variable_scope('PreprocessImage'):
#         hr = tf.decode_raw(hr, tf.uint8).reshape((1, 180, 320, 3))
#         lr = tf.decode_raw(lr, tf.uint8).reshape((1, 180, 320, 3))
#         fl = tf.decode_raw(fl, tf.float32).reshape((1, 180, 320, 4))
#         if convert:
#             pass
#         if augment:
#             pass
#             # image = tf.image.random_flip_up_down(image)
#             # image = tf.image.random_flip_left_right(image)
#
#         hr = tf.extract_image_patches(hr, [1, 68, 68, 1], [1, 36, 36, 1], [1, 1, 1, 1], 'VALID')
#         lr = tf.extract_image_patches(lr, [1, 17, 17, 1], [1, 9, 9, 1], [1, 1, 1, 1], 'VALID')
#         fl = tf.extract_image_patches(fl, [1, 17, 17, 1], [1, 9, 9, 1], [1, 1, 1, 1], 'VALID')
#         hr_shape = hr.get_shape().as_list()
#         batch = hr_shape[1] * hr_shape[2]
#         hr = tf.reshape(hr, [batch, 68, 68, 3])
#         lr = tf.reshape(lr, [batch, 17, 17, 3])
#         fl = tf.reshape(fl, [batch, 17, 17, 4])
#
#         return hr, lr, fl
#
#
# def get_data(dataset,
#              batch_size,
#              augment=False,
#              shuffle_config=None,
#              shuffle=True):
#     if not shuffle_config:
#         shuffle_config = DEFAULT_SHUFFLE_CONFIG
#
#     provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
#                                                               shuffle=shuffle,
#                                                               common_queue_capacity=2 * batch_size,
#                                                               common_queue_min=batch_size)
#     hr, lr, fl = provider.get(['hr', 'lr', 'fl'])
#     hr, lr, fl = preprocess_image(hr, lr, fl, augment)
#     hr, lr, fl = tf.train.shuffle_batch(
#         tensors=[hr, lr, fl],
#         batch_size=batch_size,
#         enqueue_many=True,
#         num_threads=shuffle_config.num_batching_threads,
#         capacity=shuffle_config.queue_capacity,
#         min_after_dequeue=shuffle_config.min_after_dequeue)
#     return InputEndpoints(
#         hr=hr,
#         lr=lr,
#         fl=fl)

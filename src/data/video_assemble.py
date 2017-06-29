import os
import logging
import argparse
import numpy as np
import tensorflow as tf
import src.cv2 as cv2
from PIL import Image
from tensorflow.contrib import slim
from src.utility.common_flags import dops

PROJECT_ROOT_PATH = './'
DEFAULT_DATASET_DIR = 'train-data/no-up'

DEFAULT_CONFIG = {
    'name': 'video_data',
    'splits': {
        'train': {
            'size': 9498,
            'shape': [[180, 320, 3], [720, 1280, 1], [180, 320, 2, 3]],
            'pattern': 'video_x4_c3_train_assemble.tfrecords'
        },
        'val': {
            'size': 32,
            'shape': [[144, 176, 3], [576, 704, 1],[144, 176, 2, 3]],
            'pattern': 'video_x4_c3_test_assemble.tfrecords'
        }
    }
}


def get_split(split_name, dataset_dir=None, config=None):
    """Returns a dataset tuple for FSNS dataset.
    Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources, by default it uses
      a predefined CNS path (see DEFAULT_DATASET_DIR).
    config: A dictionary with dataset configuration. If None - will use the
      DEFAULT_CONFIG.
    Returns:
    A `Dataset` namedtuple.
    Raises:
    ValueError: if `split_name` is not a valid train/test split.
    """
    if not dataset_dir:
        dataset_dir = os.path.join(PROJECT_ROOT_PATH, DEFAULT_DATASET_DIR)

    if not config:
        config = DEFAULT_CONFIG

    if split_name not in config['splits']:
        raise ValueError('split name %s was not recognized.' % split_name)

    logging.info('Using %s dataset split_name=%s dataset_dir=%s', config['name'], split_name, dataset_dir)

    keys_to_features = {
        'image/encode':
            tf.FixedLenFeature((), tf.string),
        'image/label':
            tf.FixedLenFeature((), tf.string),
        'image/flow':
            tf.FixedLenFeature([np.prod(config['splits'][split_name]['shape'][-1])], tf.float32),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='raw'),
    }
    items_to_handlers = {
        'image':
            slim.tfexample_decoder.Image(
                shape=config['splits'][split_name]['shape'][0],
                image_key='image/encode',
                format_key='image/format'),
        'label':
            slim.tfexample_decoder.Image(
                shape=config['splits'][split_name]['shape'][1],
                image_key='image/label',
                format_key='image/format'),
        'flow':
            slim.tfexample_decoder.Tensor(tensor_key='image/flow'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                      items_to_handlers)

    file_pattern = config['splits'][split_name]['pattern']
    if not isinstance(file_pattern, list):
        file_pattern = [file_pattern]
    file_pattern = [os.path.join(dataset_dir, fp) for fp in file_pattern]

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=config['splits'][split_name]['size'],
        shape=config['splits'][split_name]['shape'],
        items_to_descriptions=None
    )


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def im_processing(path):
    """To generate low & high super-resolution image pair"""
    '''The process version of cv2'''
    # img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)[..., 0]
    # sz = np.asarray(img.shape) / DataOptions.scale * DataOptions.scale
    # img = img[:sz[0], :sz[1]]
    # img_lr = cv2.resize(img, None, fx=1.0/DataOptions.scale, fy=1.0/DataOptions.scale, interpolation=cv2.INTER_CUBIC)

    '''The process version of PIL'''
    img = Image.open(path).convert('YCbCr').split()[0]
    round_size = np.asarray(img.size) / dops.scale * dops.scale
    img = img.crop((0, 0, round_size[0], round_size[1]))
    img_lr = img.resize(round_size / dops.scale, resample=Image.BICUBIC)
    if dops.is_up:
        img_lr = img_lr.resize(round_size, resample=Image.BICUBIC)

    img = np.asarray(img)
    img_lr = np.asarray(img_lr)

    return img, img_lr


def get_optical_flow(prvs, next):
    return cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 15, 3, 5, 1.2, 0)


def generate_data(input_file, output_file):
    file_list = np.loadtxt(input_file, dtype=np.str, delimiter='*').tolist()
    writer = tf.python_io.TFRecordWriter(output_file)

    file_list = [os.path.join(PROJECT_ROOT_PATH, f) for f in file_list]

    for i in range(1, len(file_list) - 1):
        im1, im1_lr = im_processing(file_list[i - 1])
        im2, im2_lr = im_processing(file_list[i])
        im3, im3_lr = im_processing(file_list[i + 1])

        flow1 = get_optical_flow(im1_lr, im2_lr)
        flow2 = np.zeros_like(flow1)
        flow3 = get_optical_flow(im3_lr, im2_lr)

        image = np.stack([im1_lr, im2_lr, im3_lr], axis=-1)
        label = im2
        flow = np.stack([flow1, flow2, flow3], axis=-1)

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encode': _bytes_feature(image.tobytes()),
            'image/label': _bytes_feature(label.tobytes()),
            'image/flow': _float_feature(flow.flatten().tolist())
        }))
        writer.write(example.SerializeToString())
        logging.info(str(i))

    logging.info('Image shape %s, Label shape %s' % (str(image.shape), str(label.shape)))
    logging.info('Write done!')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video assemble data generation.')
    parser.add_argument('-i', '--input',
                        dest='input_file',
                        help='The image store list.',
                        required=True)
    parser.add_argument('-o', '--output',
                        dest='output_file',
                        help='The output file path.',
                        required=True)

    args = parser.parse_args()
    generate_data(args.input_file, args.output_file)






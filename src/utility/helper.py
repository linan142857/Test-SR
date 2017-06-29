import numpy as np
import h5py
from PIL import Image
from scipy import signal
import tensorflow as tf


def rgb_gray(img):
    if isinstance(img, str):
        img = Image.open(img)
    else:
        img = Image.fromarray(img)
    if img.mode == 'RGB':
        img = img.convert('YCbCr').split()[0]
    return img


def pnsr(x, y):
    """ Calculate PSNR with tensor x and y which the batch of images
     normalized [0, 255] with the same size (H, W, C)"""
    mse = (np.subtract(x, y) ** 2).sum() / x.size
    psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))
    return psnr


def pnsr_tf(x, y):
    """Image quality metric based on maximal signal power vs. power of the noise.
    Args:
        x: the ground truth image.
        y: the predicted image.
    Returns:
        peak signal to noise ratio (PSNR)
    """
    x *= 255
    y *= 255
    mse = tf.reduce_sum(tf.square(x - y)) / tf.to_float(tf.size(x))
    psnr = 20.0 * tf.log(255.0 / tf.sqrt(mse)) / tf.log(10.0)
    return psnr
    # mse = tf.reduce_sum(tf.square(x - y)) / tf.to_float(tf.size(x))
    # return 10.0 * tf.log(1.0 / mse) / tf.log(10.0)


def load_data(path=None):
    """ Load data from h5 form in given path """
    f = h5py.File(path, 'r')
    res = dict()
    for k, v in f.items():
        res[k] = v.value
    return res


def get_minibatches_idx(n, minibatch_size, shuffle=True):
    idx_list = np.arange(n, dtype="int32")
    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = list()
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    return range(len(minibatches)), minibatches


def get_random_minibatches_idx(n, minibatch_size):
    idx_list = np.arange(n, dtype="int32")
    np.random.shuffle(idx_list)
    return idx_list[0:minibatch_size]


def back_projection(im_h, im_l, maxIter=20, kernel_size=5):

    def fspecial_gauss(size, sigma):
        x, y = np.mgrid[-size / 2 + 1:size / 2 + 1, -size / 2 + 1:size / 2 + 1]
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / g.sum()

    im_h = np.squeeze(im_h)
    im_l = np.squeeze(im_l)
    # assert(im_h.ndim == 2, 'Only allowed two dimension image.')
    # assert(im_h.size[0] - im_l.size[0] == 0 and im_h.size[1] - im_l.size[1] == 0,
    #        'The LR image should be `BICUBIC` interpolated as the same size as HR')
    p = fspecial_gauss(kernel_size, 1) ** 2
    p /= np.sum(p)
    for i in range(maxIter):
        im = Image.fromarray(im_h)
        im_down = im.resize([im.size[0] / 4, im.size[1] / 4], resample=Image.BICUBIC)
        im_up = im_down.resize(im_h.size, resample=Image.BICUBIC)
        im_up = np.asarray(im_up)
        im_diff = im_l - im_up
        im_h += signal.convolve2d(im_diff, p, mode='same')

    return im_h


def write_image(im, name='a.jpg'):
    im = np.squeeze(np.uint8(im))
    im = Image.fromarray(im, 'L')
    im.save(name)

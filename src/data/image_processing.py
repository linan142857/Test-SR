from src.utility.common_flags import *
from video_assemble import *
import src.cv2 as cv2


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
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)[..., 0]
    sz = np.asarray(img.shape) / DataOptions.scale * DataOptions.scale
    img = img[:sz[0], :sz[1]]
    img_lr = cv2.resize(img, None, fx=1.0/DataOptions.scale, fy=1.0/DataOptions.scale, interpolation=cv2.INTER_CUBIC)

    # img = Image.open(path).convert('YCbCr').split()[0]
    # sz = np.asarray(img.size) / DataOptions.scale * DataOptions.scale
    # img = img.crop((0, 0, sz[0], sz[1]))
    # img_lr = img.resize(sz / DataOptions.scale, resample=Image.BICUBIC)
    # if DataOptions.is_scale_up:
    #     img_lr = img_lr.resize(sz, resample=Image.BICUBIC)

    return img, img_lr


def get_optical_flow(prvs, next):
    return cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 15, 3, 5, 1.2, 0)


txtfile = '/home/yulin/Documents/SR/Test-SR/data/video/val_list.txt'
ROOT_PATH = '/home/yulin/Documents/SR/'
file_list = np.loadtxt(txtfile, dtype=np.str)
# writer = tf.python_io.TFRecordWriter('/home/yulin/Documents/SR/Test-SR/train-data/no-up/video_x4_c3_val_no-up.tfrecords')
for i in range(1, file_list.size - 1):
    im1, im1_lr = im_processing(ROOT_PATH+file_list[i - 1])
    im2, im2_lr = im_processing(ROOT_PATH+file_list[i])
    im3, im3_lr = im_processing(ROOT_PATH+file_list[i + 1])
    flow1 = get_optical_flow(im1_lr, im2_lr)
    flow3 = get_optical_flow(im3_lr, im2_lr)
    lr = np.stack([im1_lr, im2_lr, im3_lr], axis=-1)
    hr = np.stack([im1, im2, im3], axis=-1)
    fl = np.stack([flow1, flow3], axis=-1)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/lr': _bytes_feature(lr.tobytes()),
        'image/hr': _bytes_feature(hr.tobytes()),
        'image/fl': _bytes_feature(fl.tobytes())
    }))
    writer.write(example.SerializeToString())
    print i

writer.close()


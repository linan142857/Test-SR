import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from src import utility
from src import data
from PIL import Image
from model import Model

FLAGS = tf.app.flags.FLAGS
up_scale = 4
is_up = False


def train_operation(global_step, loss):
    '''
    Defines train operations
    :param global_step: tensor variable with shape [1]
    :param total_loss: tensor with shape [1]
    :return: two operations. Running train_op will do optimization once. Running train_ema_op
    will generate the moving average of train loss for tensorboard
    '''

    # The ema object help calculate the moving average of train loss
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
        train_ema_op = ema.apply([loss])

        lr = tf.train.exponential_decay(FLAGS.base_lr,
                                        global_step,
                                        FLAGS.stepsize,
                                        FLAGS.gamma,
                                        staircase=True)
        train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
    return train_op, train_ema_op


def validation_op(val_step, loss):
    '''
    Defines validation operations
    :param val_step: tensor with shape [1]
    :param loss: tensor with shape [1]
    :return: validation operation
    '''

    # This ema object help calculate the moving average of validation loss and error
    # ema with decay = 0.0 won't average things at all. This returns the original error
    ema = tf.train.ExponentialMovingAverage(0.0, val_step)
    ema2 = tf.train.ExponentialMovingAverage(0.95, val_step)

    val_op = tf.group(val_step.assign_add(1), ema.apply([loss]), ema2.apply([loss]))
    # loss_val = ema.average(loss)
    # loss_val_avg = ema2.average(loss)

    return val_op


def build_train_validation_graph(placeholders):
    '''
    This function builds the train graph and validation graph at the same time.
    '''
    global_step = tf.Variable(0, trainable=False, name='global_step')
    val_step = tf.Variable(0, trainable=False)

    # Logits of training data and valiation data come from the same graph. The inference of
    # validation data share all the weights with train data. This is implemented by passing
    # reuse=True to the variable scopes of train graph
    logits = make_inference(placeholders['train_input'])
    val_logits = make_inference(placeholders['val_input'], reuse=True, is_training=False)

    # Train loss
    train_loss = make_loss(logits, placeholders['train_label'])
    # Validation loss
    val_loss = make_loss(val_logits, placeholders['val_label'], is_training=False)

    train_op, train_ema_op = train_operation(global_step, train_loss)
    val_op = validation_op(val_step, val_loss)
    return train_op, train_ema_op, train_loss, val_op, val_loss, logits, val_logits


def build_validation_graph(val_data, reuse):
    '''
    This function builds the validation graph at the same time.
    '''
    val_step = tf.Variable(0, trainable=False)
    patch_size = val_data['SIZE']
    time_step = val_data['TIME_STEP']
    channel = val_data['CHANNEL']
    batch_size = FLAGS.val_batch_size

    make_up_val = val_data['ISUP'] and 1 or val_data['UPSCALE']
    init_x_size = (batch_size, time_step, patch_size, patch_size, channel)
    init_y_size = (batch_size, time_step, patch_size * make_up_val, patch_size * make_up_val, channel)
    val_input = tf.placeholder(dtype=tf.float32, shape=init_x_size, name='val_x')
    val_label = tf.placeholder(dtype=tf.float32, shape=init_y_size, name='val_y')
    placeholders = {'val_input': val_input, 'val_label': val_label}

    val_logits = make_inference(placeholders['val_input'], reuse=reuse, is_training=False)

    # Validation loss
    val_loss = make_loss(val_logits, placeholders['val_label'], is_training=False)

    # val_op = validation_op(val_step, val_loss)
    return val_loss, val_logits, placeholders


def build_test_graph(test_data, reuse):
    lr = test_data['test_data']
    hr = test_data['test_label']
    input = tf.placeholder(dtype=tf.float32, shape=lr.shape)
    logits = make_inference(input, reuse, is_training=False)
    output_shape = tuple(logits.get_shape().as_list())
    assert output_shape == hr.shape
    return {'test_data': lr,  'test_label': hr,
            'test_input': input, 'test_logits': logits}


def get_test_dataset(dataset, is_up):
    DATASET = {'city': {'img_path': '/home/yulin/Documents/SR/Test-SR/data/test/DeepSR_data/test_synthetic/city',
                        'hr_dir': 'truth', 'lr_dir': 'bicubic_x%d' % up_scale}
    }
    options = {'scale': up_scale, 'is_gray': True, 'is_up': is_up}
    channel = options.get('is_gray') and 1 or 3

    dataset = DATASET.get(dataset)
    img_path = dataset.get('img_path')
    hr_dir = dataset.get('hr_dir')
    lr_dir = dataset.get('lr_dir')

    img_list = list()
    hr_list = os.listdir(os.path.join(img_path, hr_dir))
    for name in hr_list:
        hr_path = os.path.join(img_path, hr_dir, name)
        if os.path.isfile(hr_path):
            img_list.append(name)
    img_list.sort()

    sz_hr = np.asarray(Image.open(os.path.join(img_path, hr_dir, img_list[0])).size)
    sz_lr = np.asarray(Image.open(os.path.join(img_path, lr_dir, img_list[0])).size)
    HR = np.zeros((1, len(img_list), sz_hr[1], sz_hr[0], channel))
    LR = np.zeros((1, len(img_list), sz_lr[1], sz_lr[0], channel))

    for i, name in zip(range(len(img_list)), img_list):
        img = utility.rgb_gray(os.path.join(img_path, hr_dir, name))
        img_lr = utility.rgb_gray(os.path.join(img_path, lr_dir, name))
        hr = np.asarray(img, dtype=np.float32) / 255
        lr = np.asarray(img_lr, dtype=np.float32) / 255
        lr = lr.reshape([1] + list(lr.shape) + [channel])
        hr = hr.reshape([1] + list(hr.shape) + [channel])
        HR[0, i, :, :, :] = hr
        LR[0, i, :, :, :] = lr
    return {'test_data': LR, 'test_label': HR}


def make_placeholders(train_data, val_data):
    '''
    There are four placeholders in total.
    '''

    patch_size = train_data['SIZE']
    time_step = train_data['TIME_STEP']
    channel = train_data['CHANNEL']
    batch_size = FLAGS.batch_size

    make_up = train_data['ISUP'] and 1 or train_data['UPSCALE']
    make_up_val = val_data['ISUP'] and 1 or val_data['UPSCALE']
    assert make_up_val == make_up, 'The upscale of train and validation must be the same'
    init_x_size = (batch_size, time_step, patch_size, patch_size, channel)
    init_y_size = (batch_size, time_step, patch_size * make_up, patch_size * make_up, channel)
    train_input = tf.placeholder(dtype=tf.float32, shape=init_x_size, name='x')
    train_label = tf.placeholder(dtype=tf.float32, shape=init_y_size, name='y')

    patch_size = val_data['SIZE']
    time_step = val_data['TIME_STEP']
    channel = val_data['CHANNEL']
    batch_size = FLAGS.val_batch_size

    init_x_size = (batch_size, time_step, patch_size, patch_size, channel)
    init_y_size = (batch_size, time_step, patch_size * make_up, patch_size * make_up, channel)
    val_input = tf.placeholder(dtype=tf.float32, shape=init_x_size, name='val_x')
    val_label = tf.placeholder(dtype=tf.float32, shape=init_y_size, name='val_y')
    placeholders = {
        'train_input':  train_input,
        'train_label':  train_label,
        'val_input':    val_input,
        'val_label':    val_label}

    return placeholders


def make_loss(logits, labels, is_training=True):
    # The following codes calculate the train loss, which is consist of the
    # L2 norm loss and the relularization loss
    '''
    Calculate the cross entropy loss given logits and true labels
    :param logits: 2D tensor with shape [batch_size, num_labels]
    :param labels: 1D tensor with shape [batch_size]
    :param is_training: bool decided to add regular_losse
    :return: loss tensor with shape [1]
    '''
    diff = labels - logits
    loss = tf.nn.l2_loss(diff) / tf.size(diff)
    # loss = tf.nn.l2_loss(labels - logits) / labels.get_shape().as_list()[0]
    if is_training:
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([loss] + regu_losses)
    return loss


def make_inference(x, reuse=False, is_training=True):
    '''
    The main function that defines the model.
    :param x: 5D tensor
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, reuse=True
    :param is_training: bool variable.
    :return: last layer in the network.
    '''
    model = Model(reuse=reuse, flags=FLAGS, upscale=up_scale, is_training=is_training)
    return model.bidirection_rnn(x)


def test(dataset, is_up):
    test_data = get_test_dataset(dataset, is_up)
    test_op = build_test_graph(test_data, reuse=False)

    print('Load weights from checkpoint...')
    saver = tf.train.Saver(tf.global_variables())
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(FLAGS.test_ckpt_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

    data, label = test_op['test_data'], test_op['test_label']
    input, logits = test_op['test_input'], test_op['test_logits']
    feed_dict = {input: data}
    [pred] = sess.run([logits], feed_dict=feed_dict)
    pred, label = np.squeeze(np.rint(pred * 255)), np.squeeze(np.rint(label * 255))
    pred, label = np.split(pred, pred.shape[0], axis=0), np.split(label, label.shape[0], axis=0)
    pnsr = [utility.pnsr(x, y) for x, y in zip(pred, label)]
    average_psnr = np.mean(pnsr)
    print('---------------- test stage %s: psnr = %.2f' % (datetime.now(), average_psnr))
    return average_psnr


def validation():
    val_data = utility.load_data(
        path='/home/yulin/Documents/SR/Test-SR/train-data/no-up/video_x4_c1_z17_s30_f20_n1000_no-up.h5')
    val_loss, val_logits, placeholders = build_validation_graph(val_data, reuse=False)
    saver = tf.train.Saver(tf.all_variables())
    sess = tf.Session()
    ckpt = tf.train.get_checkpoint_state(FLAGS.test_ckpt_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Load weights from checkpoint...')

    no_batch, val_batch_index = utility.get_minibatches_idx(val_data['COUNT'], FLAGS.val_batch_size, shuffle=False)
    loss_list = list()
    pnsr_list = list()
    for i, index in zip(no_batch, val_batch_index):
        val_batch_data, val_batch_label = val_data['data'][index], val_data['label'][index]
        feed_dict = {placeholders['val_input']: val_batch_data,
                     placeholders['val_label']: val_batch_label}
        [loss_value, logits_value] = sess.run([val_loss, val_logits], feed_dict=feed_dict)
        loss_list.append(loss_value)
        batch_pred = np.squeeze(np.rint(logits_value * 255))
        batch_label = np.squeeze(np.rint(val_batch_label * 255))
        shape = batch_label.shape
        batch_pred = np.reshape(batch_pred, [shape[0]*shape[1], shape[2], shape[3], -1])
        batch_label = np.reshape(batch_label, [shape[0]*shape[1], shape[2], shape[3], -1])
        pnsr = [utility.pnsr(x, y) for x, y in zip(batch_pred, batch_label)]
        pnsr_list += pnsr
    print ('-------- validation stage %s: loss = %.2f, pnsr = %.2f.'
           % (datetime.now(), np.mean(loss_list), np.mean(pnsr_list)))


def train(train_data, val_data):
    batch_size, val_batch_size = FLAGS.batch_size, FLAGS.val_batch_size
    n_samples, n_val_sample, is_up = train_data['COUNT'], val_data['COUNT'], val_data['ISUP']

    placeholders = make_placeholders(train_data, val_data)
    train_op, train_ema_op, train_loss,\
    val_op, val_loss, logits, val_logits = build_train_validation_graph(placeholders)
    test_data = get_test_dataset('city', is_up)
    test_op = build_test_graph(test_data, reuse=True)

    saver = tf.train.Saver(tf.global_variables())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    if FLAGS.is_use_ckpt is True:
        ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restored from checkpoint...')
    else:
        sess.run(tf.global_variables_initializer())

    # These lists are used to save a csv file at last
    step_train_list = []
    train_loss_list = []
    step_val_list = []
    val_loss_list = []
    step_test_list = []
    test_loss_list = []

    print('Start training...')
    print('----------------------------')
    format_str = '%s: step %d, loss = %.2f'
    for step in range(int(FLAGS.max_steps)):
        if step % FLAGS.test_interval == 0:
            data, label = test_op['test_data'], test_op['test_label']
            input, logits = test_op['test_input'], test_op['test_logits']
            feed_dict = {input: data}
            [pred] = sess.run([logits], feed_dict=feed_dict)
            pred = np.squeeze(np.rint(pred * 255))
            label = np.squeeze(np.rint(label * 255))
            pred = np.split(pred, pred.shape[0], axis=0)
            label = np.split(label, label.shape[0], axis=0)
            pnsr = [utility.pnsr(x, y) for x, y in zip(pred, label)]
            average_psnr = np.mean(pnsr)
            step_test_list.append(step)
            test_loss_list.append(average_psnr)
            print('---------------- test stage %s: step %d, psnr = %.2f' % (datetime.now(), step, average_psnr))

        if step % FLAGS.val_interval == 0:
            no_batch, val_batch_index = utility.get_minibatches_idx(n_val_sample, val_batch_size, shuffle=False)
            loss_list = []
            for i, index in zip(no_batch, val_batch_index):
                val_batch_data, val_batch_label = val_data['data'][index], val_data['label'][index]
                feed_dict = {placeholders['val_input']: val_batch_data,
                             placeholders['val_label']: val_batch_label}
                [loss_value] = sess.run([val_loss], feed_dict=feed_dict)
                loss_list.append(loss_value)
            step_val_list.append(step)
            val_loss_list.append(np.mean(loss_list))
            print (('-------- validation stage ' + format_str) % (datetime.now(), step, loss_value))

        train_batch_index = utility.get_random_minibatches_idx(n_samples, batch_size)
        train_batch_data, train_batch_label = train_data['data'][train_batch_index], train_data['label'][train_batch_index]
        feed_dict = {placeholders['train_input']: train_batch_data,
                     placeholders['train_label']: train_batch_label}
        _, loss_value = sess.run([train_op, train_loss], feed_dict=feed_dict)
        if step % FLAGS.display == 0:
            step_train_list.append(step)
            train_loss_list.append(loss_value)
            print (('train stage ' + format_str) % (datetime.now(), step, loss_value))

        # Save checkpoints
        if step % FLAGS.snapshot == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.ckpt_path, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

            df = pd.DataFrame(data={'step': step_train_list, 'train_loss': train_loss_list})
            df.to_csv(os.path.join(FLAGS.log_dir, FLAGS.version + '_train_loss.csv'))
            df = pd.DataFrame(data={'step': step_val_list, 'val_loss': val_loss_list})
            df.to_csv(os.path.join(FLAGS.log_dir, FLAGS.version + '_val_loss.csv'))
            df = pd.DataFrame(data={'step': step_test_list, 'test_psnr': test_loss_list})
            df.to_csv(os.path.join(FLAGS.log_dir, FLAGS.version + '_test_loss.csv'))


def hyper_parameters():
    model_name = 'video-bilinear_conv-K3-D10-N32-C3-assemble'
    root_path = '/home/yulin/Documents/SR/Test-SR'
    ckpt_path = os.path.join(root_path, 'model', model_name)
    log_dir = os.path.join(root_path, 'logs', model_name)
    os.path.exists(ckpt_path) or os.makedirs(ckpt_path)
    os.path.exists(log_dir) or os.makedirs(log_dir)
    tf.app.flags.DEFINE_integer('display', 100, '''Display interval''')
    tf.app.flags.DEFINE_integer('batch_size', 10, '''Train batch size''')
    tf.app.flags.DEFINE_integer('val_batch_size', 10, '''Validation batch size''')
    tf.app.flags.DEFINE_integer('stepsize', 30000, '''At which step to decay the learning rate''')
    tf.app.flags.DEFINE_integer('max_steps', 200000, 'Number of steps to run trainer')
    tf.app.flags.DEFINE_integer('val_interval', 1000, 'Number of steps interval to run validator')
    tf.app.flags.DEFINE_integer('test_interval', 1000, 'Number of steps interval to run tester')
    tf.app.flags.DEFINE_integer('snapshot', 1000, 'Number of steps interval to store model')
    tf.app.flags.DEFINE_integer('filter_number', 32, 'Number of convolution filter in Net')
    tf.app.flags.DEFINE_integer('filter_size', 3, 'The size of conv filter.')
    tf.app.flags.DEFINE_float('base_lr', 1e-4, '''Initial learning rate''')
    tf.app.flags.DEFINE_float('gamma', 0.5, '''How much to decay the learning rate each time''')
    tf.app.flags.DEFINE_float('weight_decay', 1e-4, '''scale for l2 regularization''')
    tf.app.flags.DEFINE_float('train_ema_decay', 0.995, '''The decay factor of the train error's
    moving average shown on tensorboard''')
    tf.app.flags.DEFINE_string('conv_type', 'SAME', '''Convolution type''')
    tf.app.flags.DEFINE_string('version', model_name, '''Model name''')
    tf.app.flags.DEFINE_string('root', root_path, '''Project path''')
    tf.app.flags.DEFINE_string('ckpt_path', ckpt_path, '''Checkpoint directory to restore''')
    tf.app.flags.DEFINE_string('test_ckpt_path', ckpt_path, '''Checkpoint directory to load testing''')
    tf.app.flags.DEFINE_string('log_dir', log_dir, '''Train batch size''')
    tf.app.flags.DEFINE_boolean('is_use_ckpt', False, '''Whether to load a checkpoint and continue training''')


def main(argv=None):
    print('Load train-val dataset...')
    train_data = utility.load_data(
        path='train-data/no-up/video_x4_c1_z17_s30_f10_n50000_no-up.h5')
    val_data = utility.load_data(
        path='train-data/no-up/video_x4_c1_z17_s30_f10_n1000_no-up.h5')
    train(train_data, val_data)


if __name__ == "__main__":
    hyper_parameters()
    tf.app.run()

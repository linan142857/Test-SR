# import logging
# import os
# from datetime import datetime
#
# import pandas as pd
# from PIL import Image
# from tensorflow.python.platform import flags
#
# import src.data as data
# import src.utility as utility
# from model import Model
# from ops import *
# from src.utility import common_flags
#
# FLAGS = flags.FLAGS
# common_flags.define()
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# def prepare_training_dir():
#     model_name = 'video-baseline-K3-D10-N32-C3-assemble+batchnorm'
#
#     flags.DEFINE_string('log_dir', os.path.join('logs', model_name),
#                         '''Directory where to write event logs''')
#     flags.DEFINE_string('ckpt_path', os.path.join('model', model_name),
#                         '''Checkpoint directory to restore''')
#     flags.DEFINE_string('version', model_name,
#                         '''Model name''')
#     flags.DEFINE_integer('up_scale', 4,
#                          '''Up scale''')
#
#     if not tf.gfile.Exists(FLAGS.log_dir):
#         logging.info('Create a new training log directory %s', FLAGS.log_dir)
#         tf.gfile.MakeDirs(FLAGS.log_dir)
#     else:
#         if FLAGS.ckpt_path:
#             logging.info('Create the model directory %s', FLAGS.ckpt_path)
#             tf.gfile.MakeDirs(FLAGS.ckpt_path)
#         else:
#             logging.info('Use already existing model directory %s', FLAGS.train_log_dir)
#
#
# def train_operation(global_step, loss):
#     '''
#     Defines train operations
#     :param global_step: tensor variable with shape [1]
#     :param total_loss: tensor with shape [1]
#     :return: two operations. Running train_op will do optimization once. Running train_ema_op
#     will generate the moving average of train loss for tensorboard
#     '''
#
#     # The ema object help calculate the moving average of train loss
#     # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#     # with tf.control_dependencies(update_ops):
#         # ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
#         # train_ema_op = ema.apply([loss])
#
#     lr = tf.train.exponential_decay(FLAGS.base_lr,
#                                     global_step,
#                                     FLAGS.stepsize,
#                                     FLAGS.gamma,
#                                     staircase=True)
#     train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
#     batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
#     batchnorm_updates_op = tf.group(*batchnorm_updates)
#     train_op = tf.group(train_op, batchnorm_updates_op)
#     return train_op
#
#
# def validation_op(val_step, loss):
#     '''
#     Defines validation operations
#     :param val_step: tensor with shape [1]
#     :param loss: tensor with shape [1]
#     :return: validation operation
#     '''
#
#     # This ema object help calculate the moving average of validation loss and error
#     # ema with decay = 0.0 won't average things at all. This returns the original error
#     ema = tf.train.ExponentialMovingAverage(0.0, val_step)
#     ema2 = tf.train.ExponentialMovingAverage(0.95, val_step)
#
#     val_op = tf.group(val_step.assign_add(1), ema.apply([loss]), ema2.apply([loss]))
#     # loss_val = ema.average(loss)
#     # loss_val_avg = ema2.average(loss)
#
#     batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION_VAL)
#     batchnorm_updates_op = tf.group(*batchnorm_updates)
#     val_op = tf.group(val_op, batchnorm_updates_op)
#
#     return val_op
#
#
# def build_train_validation_graph(train_data, val_data):
#     '''
#     This function builds the train graph and validation graph at the same time.
#     '''
#
#     patch_size = train_data['SIZE']
#     batch_size = FLAGS.batch_size
#
#     make_up = train_data['ISUP'] and 1 or train_data['UPSCALE']
#     make_up_val = val_data['ISUP'] and 1 or val_data['UPSCALE']
#     assert make_up_val == make_up, 'The upscale of train and validation must be the same'
#     init_x_size = (batch_size, patch_size, patch_size, train_data['data'].shape[-1])
#     init_y_size = (batch_size, patch_size * make_up, patch_size * make_up, train_data['label'].shape[-1])
#     x = tf.placeholder(dtype=tf.float32, shape=init_x_size, name='x')
#     y = tf.placeholder(dtype=tf.float32, shape=init_y_size, name='y')
#
#     patch_size = val_data['SIZE']
#     batch_size = FLAGS.val_batch_size
#
#     init_x_size = (batch_size, patch_size, patch_size, val_data['data'].shape[-1])
#     init_y_size = (batch_size, patch_size * make_up, patch_size * make_up, val_data['label'].shape[-1])
#     val_x = tf.placeholder(dtype=tf.float32, shape=init_x_size, name='val_x')
#     val_y = tf.placeholder(dtype=tf.float32, shape=init_y_size, name='val_y')
#     placeholders = {
#         'x':  x,
#         'y':  y,
#         'val_x':    val_x,
#         'val_y':    val_y}
#
#     global_step = tf.Variable(0, trainable=False, name='global_step')
#     val_step = tf.Variable(0, trainable=False, name='val_step')
#
#     # Logits of training data and valiation data come from the same graph. The inference of
#     # validation data share all the weights with train data. This is implemented by passing
#     # reuse=True to the variable scopes of train graph
#     logits = make_inference(placeholders['x'])
#     val_logits = make_inference(placeholders['val_x'], reuse=True, is_training=False)
#
#     # Train loss
#     train_loss = make_loss(logits, placeholders['y'])
#     # Validation loss
#     val_loss = make_loss(val_logits, placeholders['val_y'], is_training=False)
#
#     train_op = train_operation(global_step, train_loss)
#     val_op = validation_op(val_step, val_loss)
#
#     return placeholders, train_op, train_loss, val_op, val_loss
#
#
# def build_train_graph(train_data):
#     '''
#     This function builds the train graph.
#     '''
#
#     patch_size = train_data['SIZE']
#     batch_size = FLAGS.batch_size
#
#     make_up = train_data['ISUP'] and 1 or train_data['UPSCALE']
#     init_x_size = (batch_size, patch_size, patch_size, train_data['data'].shape[-1])
#     init_y_size = (batch_size, patch_size * make_up, patch_size * make_up, train_data['label'].shape[-1])
#     x = tf.placeholder(dtype=tf.float32, shape=init_x_size, name='x')
#     y = tf.placeholder(dtype=tf.float32, shape=init_y_size, name='y')
#     placeholders = {
#         'x': x,
#         'y': y
#     }
#
#     global_step = tf.Variable(0, trainable=False, name='global_step')
#
#     # Logits of training data and valiation data come from the same graph. The inference of
#     # validation data share all the weights with train data. This is implemented by passing
#     # reuse=True to the variable scopes of train graph
#     logits = make_inference(placeholders['x'])
#
#     # Train loss
#     train_loss = make_loss(logits, placeholders['y'])
#
#     train_op = train_operation(global_step, train_loss)
#
#     return placeholders, train_op, train_loss
#
#
# def build_test_graph(test_data, reuse):
#
#     test_op_list = list()
#     for ele in test_data:
#         lr_shape = ele['test_data'].shape
#         hr_shape = ele['test_label'].shape
#         input = tf.placeholder(dtype=tf.float32, shape=lr_shape)
#         logits = make_inference(input, reuse=reuse, is_training=False)
#         output_shape = tuple(logits.get_shape().as_list())
#         assert output_shape == hr_shape
#         batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION_VAL)
#         batchnorm_updates_op = tf.group(*batchnorm_updates)
#         test_op_list.append({'test_data': ele['test_data'],
#                              'test_label': ele['test_label'],
#                              'test_input': input,
#                              'test_logits': logits,
#                              'batchnorm': batchnorm_updates_op})
#     return test_op_list
#
#
# def get_test_dataset_by_assemble(dataset):
#     DATASET = {
#         'city':
#             {'img_path': 'data/test/DeepSR_data/test_synthetic/city',
#              'hr_dir': 'truth', 'lr_dir': 'bicubic_x%d' % up_scale}
#     }
#     dataset = DATASET.get(dataset)
#     img_path = dataset.get('img_path')
#     hr_dir = dataset.get('hr_dir')
#     lr_dir = dataset.get('lr_dir')
#
#     img_list = os.listdir(os.path.join(img_path, hr_dir))
#     test_data = list()
#     img_list.sort()
#
#     sz_hr = np.asarray(Image.open(os.path.join(img_path, hr_dir, img_list[0])).size)
#     sz_lr = np.asarray(Image.open(os.path.join(img_path, lr_dir, img_list[0])).size)
#
#     for i in range(len(img_list)-2):
#         HR = np.zeros((1, sz_hr[1], sz_hr[0], 1))
#         LR = np.zeros((1, sz_lr[1], sz_lr[0], 3))
#         name = img_list[i+1]
#         img = utility.rgb_gray(os.path.join(img_path, hr_dir, name))
#         hr = np.asarray(img, dtype=np.float32) / 255
#         HR[0, :, :, 0] = hr
#         for j in range(3):
#             name = img_list[i+j]
#             img_lr = utility.rgb_gray(os.path.join(img_path, lr_dir, name))
#             lr = np.asarray(img_lr, dtype=np.float32) / 255
#             LR[0, :, :, j] = lr
#         test_data.append({'test_data': LR, 'test_label': HR})
#
#     return test_data
#
#
# def get_test_dataset(dataset, is_up):
#     DATASET = {
#         'set14':
#             {'img_path': '/home/yulin/Documents/SR/Test-SR/data/test/Set14', 'suffix': ['.bmp']},
#         'set5':
#             {'img_path': '/home/yulin/Documents/SR/Test-SR/data/test/Set5', 'suffix': ['.bmp']},
#         'city':
#             {'img_path': '/home/yulin/Documents/SR/Test-SR/data/test/DeepSR_data/test_synthetic/city/truth2',
#              'suffix': ['.png']}
#     }
#     options = {'scale': up_scale, 'is_gray': True, 'is_up': is_up}
#     channel = options.get('is_gray') and 1 or 3
#     dataset = DATASET.get(dataset)
#     img_path = dataset.get('img_path')
#     suffix = dataset.get('suffix')
#     file_list = os.listdir(img_path)
#     file_list.sort()
#     test_data = list()
#     for index, file_name in zip(range(file_list), file_list):
#         file_path = os.path.join(img_path, file_name)
#         if os.path.isfile(file_path) and os.path.splitext(file_name)[1] in suffix:
#             img, img_lr = data.im_processing(options, file_path)
#             hr = np.asarray(img, dtype=np.float32) / 255
#             lr = np.asarray(img_lr, dtype=np.float32) / 255
#             lr = lr.reshape([1] + list(lr.shape) + [channel])
#             hr = hr.reshape([1] + list(hr.shape) + [channel])
#             test_data.append({'test_data': lr, 'test_label': hr})
#
#     return test_data
#
#
# def make_loss(logits, labels, is_training=True):
#     # The following codes calculate the train loss, which is consist of the
#     # L2 norm loss and the relularization loss
#     '''
#     Calculate the cross entropy loss given logits and true labels
#     :param logits: 2D tensor with shape [batch_size, num_labels]
#     :param labels: 1D tensor with shape [batch_size]
#     :param is_training: bool decided to add regular_losse
#     :return: loss tensor with shape [1]
#     '''
#
#     loss = l2_loss(labels, logits) / tf.size(labels, out_type=tf.float32) * tf.constant(5e3, dtype=tf.float32)
#     if is_training:
#         regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#         loss = tf.add_n([loss] + regu_losses)
#     return loss
#
#
# def make_inference(x, reuse=False, is_training=True):
#     '''
#     The main function that defines the model.
#     :param x: 4D tensor
#     :param reuse: To build train graph, reuse=False. To build validation graph and share weights
#     with train graph, resue=True
#     :param is_training: bool variable.
#     :return: last layer in the network.
#     '''
#     model = Model(reuse, FLAGS, up_scale, is_training)
#     return model.base_line_assemble_model(x)
#
#
# def test(dataset, is_up):
#     test_data = get_test_dataset(dataset, is_up)
#     test_op = build_test_graph(test_data, reuse=False)
#
#     print('Load weights from checkpoint...')
#     saver = tf.train.Saver(tf.global_variables())
#     sess = tf.Session()
#     ckpt = tf.train.get_checkpoint_state(FLAGS.test_ckpt_path)
#     saver.restore(sess, ckpt.model_checkpoint_path)
#
#     pnsr_list = list()
#     for ele in test_op:
#         data, label = ele['test_data'], ele['test_label']
#         input, logits = ele['test_input'], ele['test_logits']
#         feed_dict = {input: data}
#         [pred] = sess.run([logits], feed_dict=feed_dict)
#         pnsr = utility.pnsr(np.squeeze(np.rint(pred * 255)),
#                             np.squeeze(np.rint(label * 255)))
#         pnsr_list.append(pnsr)
#     average_psnr = np.mean(pnsr_list)
#     print('test stage %s, psnr = %.2f' % (datetime.now(), average_psnr))
#     return average_psnr
#
#
# def train(train_data, val_data):
#
#     batch_size, val_batch_size = FLAGS.batch_size, FLAGS.val_batch_size
#     n_samples, n_val_sample, is_up = train_data['COUNT'], val_data['COUNT'], val_data['ISUP']
#
#     # placeholders, train_op, train_loss, val_op, val_loss = build_train_validation_graph(train_data, val_data)
#     placeholders, train_op, train_loss = build_train_graph(train_data)
#     test_data = get_test_dataset_by_assemble('city')
#     test_op = build_test_graph(test_data, reuse=True)
#
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = tf.Session(config=config)
#     saver = tf.train.Saver(tf.global_variables())
#     if FLAGS.is_use_ckpt is True:
#         ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
#         saver.restore(sess, ckpt.model_checkpoint_path)
#         print('Restored from checkpoint...')
#     else:
#         sess.run(tf.global_variables_initializer())
#
#     # These lists are used to save a csv file at last
#     step_train_list, train_loss_list = [], []
#     step_val_list, val_loss_list = [], []
#     step_test_list, test_loss_list = [], []
#
#     print('Start training...\n----------------------------')
#     format_str = '%s: step %d, loss = %.2f'
#     for step in range(int(FLAGS.max_steps)):
#         ''' Test Stage'''
#         if step % FLAGS.test_interval == 0:
#             pnsr_list = list()
#             for ele in test_op:
#                 data, label = ele['test_data'], ele['test_label']
#                 input, logits = ele['test_input'], ele['test_logits']
#                 batchnorm = ele['batchnorm']
#                 feed_dict = {input: data}
#                 [pred] = sess.run([logits], feed_dict=feed_dict)
#                 pnsr = utility.pnsr(np.squeeze(np.rint(pred * 255)),
#                                     np.squeeze(np.rint(label * 255)))
#                 pnsr_list.append(pnsr)
#             average_psnr = np.mean(pnsr_list)
#             step_test_list.append(step)
#             test_loss_list.append(average_psnr)
#             print('---------------- test stage %s: step %d, psnr = %.2f' % (datetime.now(), step, average_psnr))
#         ''' Validation Stage'''
#         # if step % FLAGS.val_interval == 0:
#         #     no_batch, index = utility.get_minibatches_idx(n_val_sample, val_batch_size, shuffle=False)
#         #     loss_list = []
#         #     for i, inx in zip(no_batch, index):
#         #         data, label = val_data['data'][inx], val_data['label'][inx]
#         #         feed_dict = {placeholders['val_x']: data,
#         #                      placeholders['val_y']: label}
#         #         [loss_value] = sess.run([val_loss], feed_dict=feed_dict)
#         #         loss_list.append(loss_value)
#         #     step_val_list.append(step)
#         #     val_loss_list.append(np.mean(loss_list))
#         #     print (('-------- validation stage ' + format_str) % (datetime.now(), step, loss_value))
#
#         '''Train Stage'''
#         index = utility.get_random_minibatches_idx(n_samples, batch_size)
#         data, label = train_data['data'][index], train_data['label'][index]
#         feed_dict = {placeholders['x']: data,
#                      placeholders['y']: label}
#         _, loss_value = sess.run([train_op, train_loss], feed_dict=feed_dict)
#         if step % FLAGS.display == 0:
#             step_train_list.append(step)
#             train_loss_list.append(loss_value)
#             print (('train stage ' + format_str) % (datetime.now(), step, loss_value))
#
#         # Save checkpoints
#         if step % FLAGS.snapshot == 0 or (step + 1) == FLAGS.max_steps:
#             checkpoint_path = os.path.join(FLAGS.ckpt_path, 'model.ckpt')
#             saver.save(sess, checkpoint_path, global_step=step)
#
#             df = pd.DataFrame(data={'step': step_train_list, 'train_loss': train_loss_list})
#             df.to_csv(os.path.join(FLAGS.log_dir, FLAGS.version + '_train_loss.csv'))
#             # df = pd.DataFrame(data={'step': step_val_list, 'val_loss': val_loss_list})
#             # df.to_csv(os.path.join(FLAGS.log_dir, FLAGS.version + '_val_loss.csv'))
#             df = pd.DataFrame(data={'step': step_test_list, 'test_psnr': test_loss_list})
#             df.to_csv(os.path.join(FLAGS.log_dir, FLAGS.version + '_test_loss.csv'))
#
#
#
#
# def main(argv=None):
#     print('Load train-val dataset...')
#     train_data = utility.load_data(path=os.path.join(FLAGS.root_path, 'train-data/no-up/video_x4_c3_z17_s30_n500000_no-up-assemble.h5'))
#     val_data = utility.load_data(path=os.path.join(FLAGS.root_path, 'train-data/no-up/video_x4_c3_z17_s30_n1000_no-up-assemble.h5'))
#     train_data['data'] = np.transpose(train_data['data'], (0, 2, 3, 1))
#     train_data['label'] = np.transpose(train_data['label'], (0, 2, 3, 1))
#     val_data['data'] = np.transpose(val_data['data'], (0, 2, 3, 1))
#     val_data['label'] = np.transpose(val_data['label'], (0, 2, 3, 1))
#     train(train_data, val_data)
#
#
# if __name__ == "__main__":
#     tf.app.run()

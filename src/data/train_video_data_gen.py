import numpy as np
import h5py
from PIL import Image
import os
import cv2
import skvideo.io


def im_processing(options, img):
    is_gray = options.get('is_gray')
    if is_gray is None:
        is_gray = True
    is_up = options.get('is_up')
    if is_up is None:
        is_up = True
    upscale = options.get('scale') or 4
    if isinstance(img, str):
        img = Image.open(img)
    else:
        img = Image.fromarray(img)
    if is_gray and img.mode == 'RGB':
        img = img.convert('YCbCr').split()[0]
    sz = np.asarray(img.size) / upscale * upscale
    img = img.crop((0, 0, sz[0], sz[1]))
    img_lr = img.resize(sz / upscale, resample=Image.BICUBIC)
    if is_up:
        img_lr = img_lr.resize(sz, resample=Image.BICUBIC)

    return img, img_lr


def img_augmentation(im):
    fliplr = im.transpose(Image.FLIP_LEFT_RIGHT)
    flipud = im.transpose(Image.FLIP_TOP_BOTTOM)
    flipcross = fliplr.transpose(Image.FLIP_TOP_BOTTOM)
    rot90 = im.transpose(Image.ROTATE_90)
    rot270 = im.transpose(Image.ROTATE_270)
    rot90_ = fliplr.transpose(Image.ROTATE_90)
    rot270_ = fliplr.transpose(Image.ROTATE_270)
    return im, fliplr, flipud, flipcross, rot90, rot270, rot90_, rot270_


def generate_h5(options):
    video_dir = options.get('video_dir')
    suffix = options.get('suffix')
    is_gray = options.get('is_gray')
    if is_gray is None:
        is_gray = True
    is_up = options.get('is_up')
    if is_up is None:
        is_up = True
    upscale = options.get('scale') or 4
    channel = is_gray and 1 or 3
    stride = options.get('stride') or 33
    patch_sz = options.get('size') or 33
    frames = options.get('frames') or 20
    file_list = os.listdir(video_dir)
    num = int(5e4)
    count = 0
    frm = 500

    if is_up:
        HR = np.zeros((num, frames, patch_sz, patch_sz, channel)).astype(np.float32)
    else:
        HR = np.zeros((num, frames, patch_sz * upscale, patch_sz * upscale, channel)).astype(np.float32)
    LR = np.zeros((num, frames, patch_sz, patch_sz, channel)).astype(np.float32)

    for index, file_name in zip(range(len(file_list)), file_list):
        file_path = os.path.join(video_dir, file_name)
        if os.path.isfile(file_path) and os.path.splitext(file_name)[1] in suffix:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
                width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
                # height, width = cap.src_height, cap.src_width
                if is_up:
                    lr_sz = (height, width)
                else:
                    lr_sz = (height / upscale, width / upscale)
                hr = np.zeros((frames, height, width, channel)).astype(np.float32)
                lr = np.zeros((frames, lr_sz[0], lr_sz[1], channel)).astype(np.float32)
                for i in range(frm):
                    ret, frame = cap.read()
                while ret:
                    for i in range(frames):
                        img, img_lr = im_processing(options, frame)
                        img = np.asarray(img, dtype=np.float32) / 255
                        img_lr = np.asarray(img_lr, dtype=np.float32) / 255
                        img = img.reshape(list(img.shape) + [channel])
                        img_lr = img_lr.reshape(list(img_lr.shape) + [channel])
                        hr[i, :, :, :] = img
                        lr[i, :, :, :] = img_lr
                        ret, frame = cap.read()
                        frm += 1
                    for x in range(0, lr_sz[0] - patch_sz, stride):
                        for y in range(0, lr_sz[1] - patch_sz, stride):
                            lr_patch = lr[:, x:x+patch_sz, y:y+patch_sz, :]
                            if is_up:
                                hr_patch = hr[:, x:x+patch_sz, y:y+patch_sz, :]
                            else:
                                hr_patch = hr[:, x*upscale:x*upscale+patch_sz*upscale,
                                              y*upscale:y*upscale+patch_sz*upscale, :]
                            HR[count, :, :, :, :], LR[count, :, :, :, :] = hr_patch, lr_patch
                            count += 1
                            if count >= num:
                                break
                        if count >= num:
                            break
                    print('Index at No.' + str(frm) + ' Frames. Deal with `' + file_name + '` by cropping ' + str(count) + ' patches in all')
                    if count >= num:
                        break
            cap.release()
        if count >= num:
            break

    idx_list = np.arange(count, dtype=np.int32)
    np.random.shuffle(idx_list)
    HR = HR[idx_list]
    LR = LR[idx_list]
    save_path = '/home/yulin/Documents/SR/Test-SR/train-data/' + \
                (is_up and 'up/' or 'no-up/') + \
                '/video_x' + str(upscale) + '_c' + str(channel) + '_z' + str(patch_sz) + '_s' + str(stride) + \
                '_f' + str(frames) + '_n' + str(count) + '_' + (is_up and 'up' or 'no-up') + '.h5'

    f = h5py.File(save_path, 'w')
    f.create_dataset('data', data=LR)
    f.create_dataset('label', data=HR)
    f.create_dataset('CHANNEL', data=channel)
    f.create_dataset('SIZE', data=patch_sz)
    f.create_dataset('COUNT', data=count)
    f.create_dataset('UPSCALE', data=upscale)
    f.create_dataset('ISUP', data=is_up)
    f.create_dataset('TIME_STEP', data=frames)
    f.close()

if __name__ == '__main__':
    options = {'video_dir': '/home/yulin/Documents/SR/Test-SR/data/video',
               'suffix': ['.mkv'], 'is_up': False, 'size': 17, 'stride': 17, 'frames': 10}
    generate_h5(options)



















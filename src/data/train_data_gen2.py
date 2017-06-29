import numpy as np
import h5py
from PIL import Image
import os
import src.cv2 as cv2

DATASET = {'video':
               {'img_path': '/home/yulin/Documents/SR/Test-SR/data/video/img', 'suffix': ['.png']}
           }


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
    # fliplr = im.copy()[:, ::-1]
    # flipud = im.copy()[::-1, :]
    # flipcross = im.copy()[::-1, ::-1]
    return im, fliplr, flipud, flipcross, rot90, rot270, rot90_, rot270_

def generate_h5(options):

    dataset = DATASET.get(options['dataset'])
    is_up = options.get('is_up')
    if is_up is None:
        is_up = True
    is_aug = options.get('is_aug')
    if is_aug is None:
        is_aug = False
    options['is_gray'] = True
    upscale = options.get('scale') or 4
    stride = options.get('stride') or 33
    patch_sz = options.get('size') or 41
    channel = options.get('frms') or 3
    suffix = dataset.get('suffix')
    file_list = os.listdir(dataset.get('img_path'))
    file_list.sort()
    num = int(1e3)
    count = 0

    if is_up:
        HR = np.zeros((num, patch_sz, patch_sz, 1)).astype(np.float32)
    else:
        HR = np.zeros((num, patch_sz * upscale, patch_sz * upscale, 1)).astype(np.float32)
    LR = np.zeros((num, patch_sz, patch_sz, channel)).astype(np.float32)
    FL = np.zeros((num, patch_sz, patch_sz, 2, 2)).astype(np.float32)
    for i in range(8000, 10000):
        file_name = file_list[i - 1]
        file_path = os.path.join(dataset.get('img_path'), file_name)
        assert os.path.isfile(file_path) and os.path.splitext(file_name)[1] in suffix
        _, img_lr1 = im_processing(options, file_path)
        lr_1 = np.asarray(img_lr1, dtype=np.float32)
        file_name = file_list[i + 1]
        file_path = os.path.join(dataset.get('img_path'), file_name)
        assert os.path.isfile(file_path) and os.path.splitext(file_name)[1] in suffix
        _, img_lr3 = im_processing(options, file_path)
        lr_3 = np.asarray(img_lr3, dtype=np.float32)
        file_name = file_list[i]
        file_path = os.path.join(dataset.get('img_path'), file_name)
        assert os.path.isfile(file_path) and os.path.splitext(file_name)[1] in suffix
        img, img_lr2 = im_processing(options, file_path)
        lr_2 = np.asarray(img_lr2, dtype=np.float32)
        flow1 = cv2.calcOpticalFlowFarneback(lr_1, lr_2, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow3 = cv2.calcOpticalFlowFarneback(lr_3, lr_2, 0.5, 3, 15, 3, 5, 1.2, 0)

        hr = np.asarray(img, dtype=np.float32)
        if not 'lr' in locals().keys():
            lr = np.zeros(list(lr_2.shape) + [channel]).astype(np.float32)
            fl = np.zeros(list(lr_2.shape) + [2, 2]).astype(np.float32)
        lr[..., 0] = lr_1
        lr[..., 1] = lr_2
        lr[..., 2] = lr_3
        fl[..., 0] = flow1
        fl[..., 1] = flow3

        hr_sz, lr_sz = hr.shape, lr.shape
        for x in range(0, lr_sz[0] - patch_sz, stride):
            for y in range(0, lr_sz[1] - patch_sz, stride):
                lr_patch = lr[x:x+patch_sz, y:y+patch_sz, :]
                fl_patch = fl[x:x+patch_sz, y:y+patch_sz, :, :]
                if is_up:
                    hr_patch = hr[x:x+patch_sz, y:y+patch_sz]
                else:
                    hr_patch = hr[x*upscale:x*upscale+patch_sz*upscale,
                                  y*upscale:y*upscale+patch_sz*upscale]
                HR[count, :, :, 0], LR[count, :, :, :], FL[count, :, :, :, :] = hr_patch, lr_patch, fl_patch
                count += 1
                if count >= num:
                    break
            if count >= num:
                break
        if count >= num:
            break

        print('Deal with No.' + str(i) + ' Image `'
              + file_name + '` by cropping ' + str(count) + ' patches in all')

    idx_list = np.arange(count, dtype=np.int32)
    np.random.shuffle(idx_list)
    HR = np.transpose(HR[idx_list], (0, 3, 1, 2)) / 255
    LR = np.transpose(LR[idx_list], (0, 3, 1, 2)) / 255
    FL = np.transpose(FL[idx_list], (0, 4, 1, 2, 3)) / 255
    save_path = '/home/yulin/Documents/SR/Test-SR/train-data/' + \
                (is_up and 'up/' or 'no-up/') + options.get('dataset') +\
                '_x' + str(upscale) + '_c' + str(channel) + '_z' + str(patch_sz) + \
                '_s' + str(stride) + '_n' + str(count) + '_' + (is_up and 'up' or 'no-up') + '_op.h5'

    f = h5py.File(save_path, 'w')
    f.create_dataset('data', data=LR)
    f.create_dataset('opt', data=FL)
    f.create_dataset('label', data=HR)
    f.create_dataset('DATASET', data=options['dataset'])
    f.create_dataset('CHANNEL', data=channel)
    f.create_dataset('SIZE', data=patch_sz)
    f.create_dataset('COUNT', data=count)
    f.create_dataset('UPSCALE', data=upscale)
    f.create_dataset('ISUP', data=is_up)
    f.close()

if __name__ == '__main__':
    options = {'dataset': 'video', 'is_up': False, 'size': 17, 'stride': 30, 'scale': 4, 'frms': 3}
    generate_h5(options)



















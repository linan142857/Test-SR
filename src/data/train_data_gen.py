import numpy as np
import h5py
from PIL import Image
import os

DATASET = {'91':
               {'img_path': '/home/yulin/Documents/SR/Test-SR/data/91 images/', 'suffix': ['.bmp']},
           '291':
               {'img_path': '/home/yulin/Documents/SR/Test-SR/data/291/', 'suffix': ['.jpg', '.bmp']},
           'BSR':
               {'img_path': '/home/yulin/Documents/SR/Test-SR/data/BSR/', 'suffix': ['.jpg']},
           'test':
               {'img_path': '/home/yulin/Documents/SR/Test-SR/test/', 'suffix': ['.bmp']},
           'set14':
               {'img_path': '/home/yulin/Documents/SR/Test-SR/data/test/Set14', 'suffix': ['.bmp']},
           'set5':
               {'img_path': '/home/yulin/Documents/SR/Test-SR/data/test/Set5', 'suffix': ['.bmp']},
           'video':
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
    is_gray = options.get('is_gray')
    if is_gray is None:
        is_gray = True
    is_up = options.get('is_up')
    if is_up is None:
        is_up = True
    is_aug = options.get('is_aug')
    if is_aug is None:
        is_aug = False
    upscale = options.get('scale') or 4
    channel = is_gray and 1 or 3
    stride = options.get('stride') or 33
    patch_sz = options.get('size') or 41

    suffix = dataset.get('suffix')
    file_list = os.listdir(dataset.get('img_path'))
    file_list.sort(reverse=True)
    num = int(1e5)
    count = 0

    if is_up:
        HR = np.zeros((num, patch_sz, patch_sz, channel)).astype(np.float32)
    else:
        HR = np.zeros((num, patch_sz * upscale, patch_sz * upscale, channel)).astype(np.float32)
    LR = np.zeros((num, patch_sz, patch_sz, channel)).astype(np.float32)
    for index, file_name in zip(range(len(file_list)), file_list):
        file_path = os.path.join(dataset.get('img_path'), file_name)
        if os.path.isfile(file_path) and os.path.splitext(file_name)[1] in suffix:
            img, img_lr = im_processing(options, file_path)
            if is_aug:
                hr_aug = img_augmentation(img)
                lr_aug = img_augmentation(img_lr)
            else:
                hr_aug, lr_aug = [img], [img_lr]
            for hr, lr in zip(hr_aug, lr_aug):
                hr = np.asarray(hr, dtype=np.float32) / 255
                lr = np.asarray(lr, dtype=np.float32) / 255
                lr = lr.reshape(list(lr.shape) + [channel])
                hr = hr.reshape(list(hr.shape) + [channel])
                hr_sz, lr_sz = hr.shape, lr.shape
                for x in range(0, lr_sz[0] - patch_sz, stride):
                    for y in range(0, lr_sz[1] - patch_sz, stride):
                        lr_patch = lr[x:x+patch_sz, y:y+patch_sz, :]
                        if is_up:
                            hr_patch = hr[x:x+patch_sz, y:y+patch_sz, :]
                        else:
                            hr_patch = hr[x*upscale:x*upscale+patch_sz*upscale,
                                          y*upscale:y*upscale+patch_sz*upscale, :]
                        HR[count, :, :, :], LR[count, :, :, :] = hr_patch, lr_patch
                        count += 1
                        if count >= num:
                            break
                    if count >= num:
                        break
                if count >= num:
                    break
            if count >= num:
                break

        print('Deal with No.' + str(index+1) + ' Image `'
              + file_name + '` by cropping ' + str(count) + ' patches in all')

    idx_list = np.arange(count, dtype=np.int32)
    np.random.shuffle(idx_list)
    HR = np.transpose(HR[idx_list], (0, 3, 1, 2))
    LR = np.transpose(LR[idx_list], (0, 3, 1, 2))
    save_path = '/home/yulin/Documents/SR/Test-SR/train-data/' + \
                (is_up and 'up/' or 'no-up/') + options.get('dataset') +\
                '_x' + str(upscale) + '_c' + str(channel) + '_z' + str(patch_sz) + \
                '_s' + str(stride) + '_n' + str(count) + '_' + (is_up and 'up' or 'no-up') + '.h5'

    f = h5py.File(save_path, 'w')
    f.create_dataset('data', data=LR)
    f.create_dataset('label', data=HR)
    f.create_dataset('DATASET', data=options['dataset'])
    f.create_dataset('CHANNEL', data=channel)
    f.create_dataset('SIZE', data=patch_sz)
    f.create_dataset('COUNT', data=count)
    f.create_dataset('UPSCALE', data=upscale)
    f.create_dataset('ISUP', data=is_up)
    f.close()

if __name__ == '__main__':
    options = {'dataset': 'video', 'is_up': False, 'size': 17, 'stride': 17, 'scale': 4}
    generate_h5(options)



















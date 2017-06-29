try:
    import Image
except ImportError:
    from PIL import Image
import numpy as np
import os

DATASET = '/home/yulin/Documents/SR/Test-SR/data/test/DeepSR_data/test_synthetic'
dirs = os.listdir(DATASET)

options = {'scale': 4, 'is_gray': False, 'suffix': 'png'}

for seq in dirs:
    seq_path = os.path.join(DATASET, seq, 'truth')
    hr_list = os.listdir(seq_path)
    save_path = os.path.join(DATASET, seq, 'bicubic_x%d' % options['scale'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for name in hr_list:
        img_path = os.path.join(seq_path, name)
        img = Image.open(img_path)
        sz = np.asarray(img.size) / options['scale']
        img_lr = img.resize(sz, resample=Image.BICUBIC)
        img_lr.save(os.path.join(save_path, name))
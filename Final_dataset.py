import os
from skimage import io
from torch.utils.data import Dataset
import glob
import scipy.io as sio

import torch
import torch.nn.functional as F

import warnings

warnings.filterwarnings("ignore")


class myDataset(Dataset):
    def __init__(self, i_dir, t_dir, rgb_dir, trans=None, if_test=False, from_memory=False):
        self.from_memory = from_memory
        self.load_fin = False
        self.images = []
        self.targets = []

        self.i_dir = i_dir
        self.t_dir = t_dir
        self.trans = trans

        mat = sio.loadmat(rgb_dir)
        self.rgb = mat['rgbMean'].reshape(1, 1, 3)

        img_name = os.path.join(self.i_dir, '*.jpg')
        self.f_list = glob.glob(img_name)
        self.ds_len = len(self.f_list)
        self.if_test = if_test

    def __len__(self):
        return self.ds_len

    def __getitem__(self, idx):

        if (not self.from_memory) or (not self.load_fin):
            img_name = self.f_list[idx]
            image = io.imread(img_name)
            image = image / 255. - self.rgb

            (f_path, f_name) = os.path.split(img_name)
            (name, extension) = os.path.splitext(f_name)

            mat_dir = os.path.join(self.t_dir, '%s.mat' % (name))
            mat = sio.loadmat(mat_dir)

            if self.from_memory:
                self.images.append(image)
                self.targets.append(mat)
                if len(self.images) == self.ds_len:
                    self.load_fin = True

        else:
            image = self.images[idx]
            mat = self.targets[idx]

        if not self.if_test:
            target = mat['crop_gtdens']
            sample = {'image': image, 'target': target}
            if self.trans:
                sample = self.trans(sample)

            sample['image'], sample['target'] = get_pad(sample['image']), get_pad(sample['target'])
        else:
            target = mat['all_num']
            sample = {'image': image, 'target': target}
            if self.trans:
                sample = self.trans(sample)
            sample['density_map'] = torch.from_numpy(mat['density_map'])

            sample['image'], sample['density_map'] = get_pad(sample['image']), get_pad(sample['density_map'],)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'target': torch.from_numpy(target)}


def get_pad(inputs):
    h, w = inputs.size()[-2:]
    ph, pw = (64 - h % 64), (64 - w % 64)

    if (ph != 64) or (pw != 64):
        tmp_pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]
        inputs = F.pad(inputs, tmp_pad)

    return inputs
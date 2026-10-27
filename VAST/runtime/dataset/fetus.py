from copy import deepcopy
import h5py
import math
import json
import numpy as np
import os
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from skimage.color import rgb2gray
from torchvision import transforms
from dataset.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box


class FETUSSemiDataset(Dataset):
    def __init__(self, json_file_path, mode, size=None, n_sample=None):
        self.json_file_path = json_file_path
        self.mode = mode
        self.size = size
        self.n_sample = n_sample

        if mode == 'train_l' or mode == 'train_u':
            with open(self.json_file_path, mode='r') as f:
                self.case_list = json.load(f)
            if mode == 'train_l' and n_sample is not None:
                self.case_list *= math.ceil(n_sample / len(self.case_list))
                self.case_list = self.case_list[:n_sample]
        else:
            with open(self.json_file_path, mode='r') as f:
                self.case_list = json.load(f)

    def __getitem__(self, item):
        case = self.case_list[item]

        if self.mode == 'valid':
            image_h5_file, label_h5_file = case['image'], case['label']
            with h5py.File(image_h5_file, mode='r') as f:
                image = f['image'][:]
                image = rgb2gray(image)
                image_view_id = f['view'][:]
            with h5py.File(label_h5_file, mode='r') as f:
                mask = f['mask'][:]
                label = f['label'][:]
            return (torch.from_numpy(image).unsqueeze(0).float(),
                    torch.from_numpy(image_view_id).long() - 1,
                    torch.from_numpy(mask).long(),
                    torch.from_numpy(label).long())

        elif self.mode == 'train_l':
            image_h5_file, label_h5_file = case['image'], case['label']
            with h5py.File(image_h5_file, mode='r') as f:
                image = f['image'][:]
                image = rgb2gray(image)
                image_view_id = f['view'][:]
            with h5py.File(label_h5_file, mode='r') as f:
                mask = f['mask'][:]
                label = f['label'][:]
            # basic augmentation
            #if random.random() > 0.5:
             #   image, mask = random_rot_flip(image, mask)
            #elif random.random() > 0.5:
             #   image, mask = random_rotate(image, mask)
            # resize
            x, y = image.shape
            image = zoom(image, (self.size / x, self.size / y), order=1)
            mask = zoom(mask, (self.size / x, self.size / y), order=0)

            from dataset.transform import apply_l_aug
            image, mask = apply_l_aug(image, mask)

            return (torch.from_numpy(image).unsqueeze(0).float(),
                    torch.from_numpy(image_view_id).long() - 1,
                    torch.from_numpy(mask).long(),
                    torch.from_numpy(label).long())

        elif self.mode == 'train_u':
            image_h5_file = case['image']
            with h5py.File(image_h5_file, mode='r') as f:
                image = f['image'][:]
                image = rgb2gray(image)
                image_view_id = f['view'][:]

            x, y = image.shape
            image = zoom(image, (self.size / x, self.size / y), order=1)

            # basic augmentation
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
            if random.random() > 0.5:
                image = np.flipud(image).copy()

            image_w_t = torch.from_numpy(image).unsqueeze(0).float()

            from dataset.transform import apply_u_strong_aug

            image_s1 = apply_u_strong_aug(image)
            image_s2 = apply_u_strong_aug(image)

            image_s1_t = torch.from_numpy(image_s1).unsqueeze(0).float()
            image_s2_t = torch.from_numpy(image_s2).unsqueeze(0).float()

            cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
            cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)

            return image_w_t, torch.from_numpy(image_view_id).long() - 1, image_s1_t, image_s2_t, cutmix_box1, cutmix_box2



    def __len__(self):
        return len(self.case_list)


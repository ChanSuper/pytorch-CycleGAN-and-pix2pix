from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import os.path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import random


class Parsing_dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        assert(self.A_size == self.B_size)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        index_A = index % self.A_size
        A_path = self.A_paths[index_A]
        B_path = self.B_paths[index_A]

        print(A_path) #
        print(B_path) #

        # A
        A_img = Image.open(A_path).convert('RGB')
        print(A_img.size) #
        A_img = scale_width(A_img, self.opt.loadSize)
        A_img = np.array(A_img)
        print(A_img.shape) #

        # B
        B_img = Image.open(B_path)
        print(B_img.size) #
        B_img = scale_width(B_img, self.opt.loadSize)
        print(B_img.size) #
        B_array_channel1 = np.array(B_img)
        B_array_channelk = np.zeros((self.opt.parts, B_array_channel1.shape[0], B_array_channel1.shape[1]), dtype=np.float32)
        for i in range(self.opt.parts):
            B_array_channelk[i] = (B_array_channel1 == i).astype(np.float32)
        print(B_array_channelk.shape)  #
        B_img = B_array_channelk.transpose((1,2,0))
        print(B_img.shape) #

        # crop
        w, h = A_img.shape[0], A_img.shape[1]
        th, tw = self.opt.fineSize, self.opt.fineSize
        print(th, tw) #
        if not (w == tw and h == th):
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            A_img = A_img[y1:y1+th, x1:x1+tw, :]
            B_img = B_img[y1:y1+th, x1:x1+tw, :]

        print(A_img.shape)
        print(B_img.shape)

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)
        # # flip
        # if not self.opt.no_flip and random.random() < 0.5:
        #     A_img = A_img.transpose(Image.FLIP_LEFT_RIGHT)
        #     B_img = B_img.transpose(Image.FLIP_LEFT_RIGHT)
        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'parsingDataset'

def scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
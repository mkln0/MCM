from __future__ import division

import os
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from PIL import Image
from utils import *
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


import pdb

class ResDatasetEva(Dataset):
    def __init__(self, path='data/valid.txt', img_size=512):
        super(ResDatasetEva, self).__init__()

        with open(path, "r") as file:
            self.img_label = [line.strip('\n') for line in file.readlines()]
        self.img_size = img_size

    def __getitem__(self, index):
        img_path, date, wd, jd, tp = self.img_label[index].split('$+$')
        img = transforms.ToTensor()(Image.open(img_path).convert(RGB))
        img, _ = pad_to_square(img, 0)
        img = resize(img,self.img_size)
        
        date = eval(date)
        wd = eval(wd)
        jd = eval(jd)
        tp = eval(tp)
        
        oth = torch.tensor([date, wd, jd])

        return img, tp, oth

    def __len__(self):
        return len(self.img_label)


class ResDataset(Dataset):
    def __init__(self, path='data/train.txt', img_size=512):
        super(ResDataset, self).__init__()

        with open(path, "r") as file:
            self.img_label = [line.strip('\n') for line in file.readlines()]
        self.img_size = img_size

    def __getitem__(self, index):
        img_path, date, wd, jd, tp = self.img_label[index].split('$+$')
        img = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        img, _ = pad_to_square(img, 0)
        img = resize(img,self.img_size)
        
        date = eval(date)
        wd = eval(wd)
        jd = eval(jd)
        tp = eval(tp)
        
        oth = torch.tensor([date, wd, jd])

        return img, tp, oth

    def __len__(self):
        return len(self.img_label)

class DetectDataset(Dataset):
    def __init__(self, path='data/samples', img_size=512):
        super(DetectDataset, self).__init__()

        self.names = os.listdir(path)
        self.img_pathes = [os.path.join(path,name) for name in self.names]
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.img_pathes[index]
        img = transforms.ToTensor()(Image.open(img_path))
        img, _ = pad_to_square(img, 0)
        img = resize(img, self.img_size)

        return img, self.names[index]

    def __len__(self):
        return len(self.img_pathes)

if __name__ == '__main__':
    dataset = DetectDataset()
    dataloader = DataLoader(dataset=dataset,batch_size=1,shuffle=True)
    for i, (image, name) in enumerate(dataloader):
        image = np.array(image[0]).transpose(1, 2, 0)
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        print(name)
        plt.show()

        
    
        

        

        

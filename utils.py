import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

import pdb



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def evaluation(net, dataloader, device, batch_size):
    net.eval()
    correct = 0
    for i, (images, labels, oths) in enumerate(dataloader):
        inputs = Variable(images.to(device)).float()
        inputs2 = Variable(oths.to(device)).float()
        labels = labels.float()
        

        outputs = net(inputs, inputs2)

        for i in range(len(labels)):
            if(torch.max(outputs.cpu().data, 1)[1][i] == labels[i]):
                correct += 1
    return correct / (len(dataloader) * batch_size)

def getSen(net, dataloader, device, batch_size):
    net.eval()
    results = []
    for i, (images, labels, oths) in enumerate(dataloader):
        inputs = Variable(images.to(device)).float()
        inputs2 = Variable(oths.to(device)).float()
        labels = labels.float()
        

        outputs = net(inputs, inputs2)
        
        outputn = outputs.cpu().data
        
        for i, (oth, oup) in enumerate(zip(oths, outputn)):
            item = torch.cat((oth, oup), 0)
            results.append(item.numpy())
        
        
        
    return np.array(results)

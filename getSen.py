from __future__ import division

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ResDataset
from smodule import SmallNet
from utils import *

import matplotlib.pyplot as plt

import math

import argparse

import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/valid.txt", help="data path")
    parser.add_argument("--batch_size", type=int, default=4, help="number of a echo train's pictures")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cup")
    parser.add_argument("--gpu_id", type=int, default=0, help="The id of gup")
    opt = parser.parse_args()
    print(opt)

    torch.cuda.set_device(opt.gpu_id)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = ResDataset(opt.data_path)
    dataLoader = DataLoader(dataset,
                            opt.batch_size,
                            shuffle=True,
                            num_workers=opt.n_cpu,
                            pin_memory=True,
                            )

    net = SmallNet()
    net.float().to(device)


    net.load_state_dict(torch.load("checkpoints/smallNet_ckpt_199.pth"))
    results = getSen(net, dataLoader, device, opt.batch_size)
    
    x1 = []
    x2 = []
    x3 = []
    
    y1 = []
    y2 = []
    
    for item in results:
        x1.append(item[0])
        x2.append(item[1])
        x3.append(item[2])
        y1.append(math.pow(math.e, item[3]))
        y2.append(math.pow(math.e, item[4]))
    
    plt.figure()
    plt.xlabel('time')
    plt.ylabel('likelihood')
    #plt.title('map curve of neural network training model for 8-stackedHourglass')

    #plt.plot(x1[:100], y1[:100])
    plt.scatter(x1, y1, c='k', marker='.')
    #ax1.scatter(x,y,c='r',marker='>')

    plt.savefig("./dataMap/time.jpg")

    plt.show()
    
    plt.figure()
    plt.xlabel('wd')
    plt.ylabel('likelihood')
    #plt.title('map curve of neural network training model for 8-stackedHourglass')

    #plt.plot(x1[:100], y1[:100])
    plt.scatter(x2, y1, c='k', marker='.')
    #ax1.scatter(x,y,c='r',marker='>')

    plt.savefig("./dataMap/wd.jpg")

    plt.show()
    
    plt.figure()
    plt.xlabel('jd')
    plt.ylabel('likelihood')
    #plt.title('map curve of neural network training model for 8-stackedHourglass')

    #plt.plot(x1[:100], y1[:100])
    plt.scatter(x3, y1, c='k', marker='.')
    #ax1.scatter(x,y,c='r',marker='>')

    plt.savefig("./dataMap/jd.jpg")

    plt.show()


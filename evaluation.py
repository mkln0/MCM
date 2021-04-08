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
    fp = open("./print_mAP.txt", "w")

    
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


    for i in range(200):
        net.load_state_dict(torch.load("checkpoints/smallNet_ckpt_" + str(i) + ".pth"))
        mean_AP = evaluation(net, dataLoader, device, opt.batch_size)
        fp.write('E{}->{}\n'.format(i, mean_AP))
        fp.flush()
    fp.close()

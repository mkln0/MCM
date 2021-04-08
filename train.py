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
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.00001, help="learning rate is a optimizer's paramenter")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum factor is a optimizer's paramenter")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="over fitting decay paramenter")
    parser.add_argument("--batch_size", type=int, default=4, help="number of a echo train's pictures")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cup")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--gpu_id", type=int, default=0, help="The id of gup")
    opt = parser.parse_args()
    print(opt)

    torch.cuda.set_device(opt.gpu_id)
    fp = open("./print_loss.txt", "w")

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = SmallNet()
    net.float().to(device)
    net.apply(weights_init_normal)
    
    dataset = ResDataset()
    dataLoader = DataLoader(dataset,
                            opt.batch_size,
                            shuffle=True,
                            num_workers=opt.n_cpu,
                            pin_memory=True,
                            )

    optimizer = optim.Adam(net.parameters(),lr=opt.lr)
    
    '''
    optimizer = optim.SGD(net.parameters(),
                          lr=opt.lr,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay,
                          )
    '''

    criterion = nn.CrossEntropyLoss()

    for epoch in range(opt.epochs):
        net.train()
        for i, (img, label, oth) in enumerate(dataLoader):
            img, label = Variable(img).to(device), Variable(label).to(device)
            oth = Variable(oth).to(device)

            output = net(img, oth)
            loss = criterion(output, label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            fp.write('[ Epoch {:005d} -> {:005d}] loss : {:15}\n'.format(
                epoch+1,
                (i+1) * opt.batch_size,
                loss.cpu().data.numpy()))
            fp.flush()

        if epoch % opt.checkpoint_interval == 0:
            torch.save(net.state_dict(), f"checkpoints/smallNet_ckpt_%d.pth" % epoch)
    

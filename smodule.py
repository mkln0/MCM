from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class SmallNet(nn.Module):
    def __init__(self, img_size = 512):
        super(SmallNet, self).__init__()

        self.img_size = img_size

        self.conv1 = nn.Conv2d(3,16,7,1,3)
        self.pool2 = nn.MaxPool2d(4,4)
        self.conv2 = nn.Conv2d(16,1,15,1,7)
        self.pool4 = nn.MaxPool2d(8,8)

        self.ln1 = nn.Linear(img_size // 32 * img_size // 32 + 3, 16)
        self.ln2 = nn.Linear(16, 2)
    def forward(self, x, oth):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool4(x)

        x = x.view(-1, self.img_size // 32 * self.img_size // 32)
        x = torch.cat((x, oth), 1)

        x = self.ln1(x)
        x = F.relu(x)
        x = self.ln2(x)
        x = F.log_softmax(x, dim=1)
        return x

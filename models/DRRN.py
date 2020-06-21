import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
import math
import pdb
import time
import numpy as np
from math import sqrt
import argparse
from torch import nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvLayer, self).__init__()
        self.module = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        )

    def forward(self, x):
        return self.module(x)


class ResidualUnit(nn.Module):
    def __init__(self, num_features):
        super(ResidualUnit, self).__init__()
        self.module = nn.Sequential(
            ConvLayer(num_features, num_features),
            ConvLayer(num_features, num_features)
        )

    def forward(self, h0, x):
        return h0 + self.module(x)


class RecursiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, U):
        super(RecursiveBlock, self).__init__()
        self.U = U
        self.h0 = ConvLayer(in_channels, out_channels)
        self.ru = ResidualUnit(out_channels)

    def forward(self, x):
        h0 = self.h0(x)
        x = h0
        for i in range(self.U):
            x = self.ru(h0, x)
        return x


class DRRN(nn.Module):
    def __init__(self, B=9, U=1, num_channels=3, num_features=128):
        super(DRRN, self).__init__()
        self.rbs = nn.Sequential(*[RecursiveBlock(num_channels if i == 0 else num_features, num_features, U) for i in range(B)])
        self.rec = ConvLayer(num_features, num_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        x = self.rbs(x)
        x = self.rec(x)
        x += residual
        return x
class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def count_parameters(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total parameters:" + str(k))

def runing_time(net, x):
    net = net.cuda()
    
    x = Variable(x.cuda())
    y = net(x)
    timer = Timer()
    timer.tic()
    for i in range(100):
        timer.tic()
        y = net(x)
        timer.toc()

    print('Do once forward need {:.3f}ms '.format(timer.total_time*1000/100.0))
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument("--scale", type=int, default=2, help="scale size")


torch.cuda.set_device(0)
if __name__ == '__main__':
    opt = parser.parse_args()
    scale = opt.scale
    x = torch.rand(1,3,48 * scale,48 * scale)
    net = DRRN()
    count_parameters(net)
    runing_time(net, x)
    
    


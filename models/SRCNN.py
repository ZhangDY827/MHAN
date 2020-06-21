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
class Net(torch.nn.Module):
    def __init__(self, num_channels = 3, base_filter = 128, upscale_factor=2):
        super(Net, self).__init__()

        self.layers = torch.nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=9, stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter, out_channels=base_filter , kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter , out_channels=num_channels , kernel_size=5, stride=1, padding=2, bias=True)
        )

    def forward(self, x):
        out = self.layers(x)
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        
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
parser.add_argument("--scale", type=int, default=4, help="scale size")


if __name__ == '__main__':
    opt = parser.parse_args()
    scale = opt.scale
    x = torch.rand(1,3,100*opt.scale,100*opt.scale)
    net = Net().cuda()
    x = x.cuda()
    t0 = time.time()
    for i in range(30):
        out = net(x)
    t = time.time() - t0
    print('average running time: ', t/30)
    count_parameters(net)
    #runing_time(net, x)
    
    


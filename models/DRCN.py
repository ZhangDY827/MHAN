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
    def __init__(self, device, num_channels = 3, base_channel = 256, num_recursions = 16):
        super(Net, self).__init__()
        self.num_recursions = num_recursions
        self.embedding_layer = nn.Sequential(
            nn.Conv2d(num_channels, base_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv_block = nn.Sequential(nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(inplace=True))

        self.reconstruction_layer = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(base_channel, num_channels, kernel_size=3, stride=1, padding=1)
        )

        self.w_init = torch.ones(self.num_recursions) / self.num_recursions
        self.w = self.w_init.to(device)

    def forward(self, x):
        h0 = self.embedding_layer(x)

        h = [h0]
        for d in range(self.num_recursions):
            h.append(self.conv_block(h[d]))

        y_d_ = list()
        out_sum = 0
        for d in range(self.num_recursions):
            y_d_.append(self.reconstruction_layer(h[d+1]))
            out_sum += torch.mul(y_d_[d], self.w[d])
        out_sum = torch.mul(out_sum, 1.0 / (torch.sum(self.w)))

        final_out = torch.add(out_sum, x)

        return y_d_, final_out

    def weight_init(self):
        for m in self._modules:
            weights_init_kaiming(m)


def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
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
parser.add_argument("--scale", type=int, default=2, help="scale size")


torch.cuda.set_device(2)
if __name__ == '__main__':
    opt = parser.parse_args()
    device = torch.device('cuda')
    scale = opt.scale
    x = torch.rand(1,3,48 * scale,48 * scale)
    net = Net(device)
    count_parameters(net)
    runing_time(net, x)
    
    


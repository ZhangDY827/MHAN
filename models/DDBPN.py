import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F
import math
import pdb
import time

import argparse

def make_model(args, parent=False):
    return DDBPN(args)

def projection_conv(in_channels, out_channels, scale, up=True):
    kernel_size, stride, padding = {
        2: (6, 2, 2),
        4: (8, 4, 2),
        8: (12, 8, 2)
    }[scale]
    if up:
        conv_f = nn.ConvTranspose2d
    else:
        conv_f = nn.Conv2d

    return conv_f(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding
    )

class DenseProjection(nn.Module):
    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        if bottleneck:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)
            ])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels

        self.conv_1 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])
        self.conv_2 = nn.Sequential(*[
            projection_conv(nr, inter_channels, scale, not up),
            nn.PReLU(inter_channels)
        ])
        self.conv_3 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        a_0 = self.conv_1(x)
        b_0 = self.conv_2(a_0)
        e = b_0.sub(x)
        a_1 = self.conv_3(e)

        out = a_0.add(a_1)

        return out

class DDBPN(nn.Module):
    def __init__(self, args):
        super(DDBPN, self).__init__()
        scale = args.scale

        n0 = 128
        nr = 32
        self.depth = 25

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        #self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        initial = [
            nn.Conv2d(args.n_colors, n0, 3, padding=1),
            nn.PReLU(n0),
            nn.Conv2d(n0, nr, 1),
            nn.PReLU(nr)
        ]
        self.initial = nn.Sequential(*initial)

        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        channels = nr
        for i in range(self.depth):
            self.upmodules.append(
                DenseProjection(channels, nr, scale, True, i > 1)
            )
            if i != 0:
                channels += nr
        
        channels = nr
        for i in range(self.depth - 1):
            self.downmodules.append(
                DenseProjection(channels, nr, scale, False, i != 0)
            )
            channels += nr

        reconstruction = [
            nn.Conv2d(self.depth * nr, args.n_colors, 3, padding=1) 
        ]
        self.reconstruction = nn.Sequential(*reconstruction)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

        #self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        #x = self.sub_mean(x)
        x = self.initial(x)

        h_list = []
        l_list = []
        for i in range(self.depth - 1):
            if i == 0:
                l = x
            else:
                l = torch.cat(l_list, dim=1)
            h_list.append(self.upmodules[i](l))
            l_list.append(self.downmodules[i](torch.cat(h_list, dim=1)))
        
        h_list.append(self.upmodules[-1](torch.cat(l_list, dim=1)))
        out = self.reconstruction(torch.cat(h_list, dim=1))
        #out = self.add_mean(out)

        return out

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

def runing_time(net):
    net = net.cuda()
    x = torch.rand(1,3,48,48)
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
parser.add_argument("--n_colors", type=int, default=3, help="scale size")

torch.cuda.set_device(1)
if __name__ == '__main__':
    opt = parser.parse_args()
    
    net = DDBPN(opt).cuda()
    x = torch.rand(1,3,100,100)
    x = x.cuda()
    t0 = time.time()
    for i in range(30):
        out = net(x)
    t = time.time() - t0
    print('Average time : ', t/30)
    count_parameters(net)
    #runing_time(net)
    
    


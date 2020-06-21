import torch
import torch.nn as nn
from math import sqrt
from torch.autograd import Variable
import torch.nn.functional as F
import time
class Net(nn.Module):
    def __init__(self, nfeats = 32):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(nfeats*3, nfeats*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv7 = nn.Conv2d(nfeats*2, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        residual = x
        im1 = self.relu(self.conv1(x))
        im2 = self.relu(self.conv2(im1))
        im3 = self.relu(self.conv3(im2))
        im4 = self.relu(self.conv4(im3))
        im5 = self.relu(self.conv5(im4))
        out = self.relu(self.conv6(torch.cat((im3, im4, im5), dim = 1)))
        out = self.conv7(out) + residual
        return out
def count_parameters(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total parameters:" + str(k))
import argparse
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument("--scale", type=int, default=4, help="scale size")


#torch.cuda.set_device(2)
if __name__ == '__main__':
    opt = parser.parse_args()
    scale = opt.scale
    x = torch.rand(1,3,100 * scale,100 * scale)
    net = Net().cuda()
    x = x.cuda()
    t0 = time.time()
    for i in range(30):
        out = net(x)
    t = time.time() - t0
    print('average running time: ', t/30)
    count_parameters(net)
    print(out.shape)
    #runing_time(net, x)

import torch
import torch.nn as nn
from math import sqrt
from torch.autograd import Variable
import torch.nn.functional as F

class D_Block(nn.Module):
    def __init__(self, nfeats = 64):
        super(D_Block, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.ud1_1 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        self.ud1_2 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        self.ud1_3 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.ud1_4 = nn.Conv2d(nfeats*4, nfeats, kernel_size=1, stride=1, padding=0, bias=False)
        self.ud1_5 = nn.Conv2d(nfeats*4, nfeats, kernel_size=1, stride=1, padding=0, bias=False)
        self.ud1_6 = nn.Conv2d(nfeats*4, nfeats, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.ud2_1 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        self.ud2_2 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        self.ud2_3 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.ud2_4 = nn.Conv2d(nfeats*4, nfeats, kernel_size=1, stride=1, padding=0, bias=False)
        self.ud2_5 = nn.Conv2d(nfeats*4, nfeats, kernel_size=1, stride=1, padding=0, bias=False)
        self.ud2_6 = nn.Conv2d(nfeats*4, nfeats, kernel_size=1, stride=1, padding=0, bias=False)

        self.ud3_1 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        self.ud3_2 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        self.ud3_3 = nn.Conv2d(nfeats, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.ud3_4 = nn.Conv2d(nfeats*4, nfeats, kernel_size=1, stride=1, padding=0, bias=False)
        self.ud3_5 = nn.Conv2d(nfeats*4, nfeats, kernel_size=1, stride=1, padding=0, bias=False)
        self.ud3_6 = nn.Conv2d(nfeats*4, nfeats, kernel_size=1, stride=1, padding=0, bias=False)
        
        
        self.ud7 = nn.Conv2d(nfeats*3, nfeats, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        x1=x2=x3=x
        a1 = self.lrelu(self.ud1_1(x1))
        b1 = self.lrelu(self.ud1_2(x2))
        c1 = self.lrelu(self.ud1_3(x3))
        
        sum = torch.cat([a1,b1,c1],1)
        x1 = self.lrelu(self.ud1_4(torch.cat([sum,x1], dim = 1)))
        x2 = self.lrelu(self.ud1_5(torch.cat([sum,x2], dim = 1)))
        x3 = self.lrelu(self.ud1_6(torch.cat([sum,x3], dim = 1)))
        
        a1 = self.lrelu(self.ud2_1(x1))
        b1 = self.lrelu(self.ud2_2(x2))
        c1 = self.lrelu(self.ud2_3(x3))
        
        sum = torch.cat([a1,b1,c1],1)
        x1 = self.lrelu(self.ud2_4(torch.cat([sum,x1], dim = 1)))
        x2 = self.lrelu(self.ud2_5(torch.cat([sum,x2], dim = 1)))
        x3 = self.lrelu(self.ud2_6(torch.cat([sum,x3], dim = 1)))
        
        a1 = self.lrelu(self.ud3_1(x1))
        b1 = self.lrelu(self.ud3_2(x2))
        c1 = self.lrelu(self.ud3_3(x3))
        
        sum = torch.cat([a1,b1,c1],1)
        x1 = self.lrelu(self.ud3_4(torch.cat([sum,x1], dim = 1)))
        x2 = self.lrelu(self.ud3_5(torch.cat([sum,x2], dim = 1)))
        x3 = self.lrelu(self.ud3_6(torch.cat([sum,x3], dim = 1)))
        
        block_out = self.lrelu(self.ud7(torch.cat([x1,x2,x3], dim = 1)))
        
        return x + block_out

class Net(nn.Module):
    def __init__(self, scale, nfeats = 64):
        super(Net, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.input2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        layers = [D_Block() for i in range(6)]
        self.res_feat1 = nn.Sequential(*layers)
        self.scale = scale
        if scale == 4:
            self.upscale_1 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
            )
        elif scale == 2:
            self.upscale_1 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
            )
        elif scale == 8:
            self.upscale_1 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
            )
        if scale == 4:
            self.res_feat2 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif scale == 2:
            self.res_feat2 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif scale == 8:
            self.res_feat2 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )
        layers = [D_Block() for i in range(3)]
        layers.append(nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.res_feat3 = nn.Sequential(*layers)
        
        self.res_feat4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.reduce = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        if scale == 4:
            self.upscale_2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
            )
        elif scale == 2:
            self.upscale_2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
            )
        elif scale == 8:
            self.upscale_2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
            )
    def Laplacian(self, x):
        kernel= torch.FloatTensor([
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[8.,0.,0.],[0.,8.,0.],[0.,0.,8.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]],
        [[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]],[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]]
        ])
        b = torch.FloatTensor([0.,0.,0.])
        weight = nn.Parameter(data=kernel, requires_grad=False)
        b = nn.Parameter(data=b, requires_grad=False)
        weight = weight.cuda()
        b = b.cuda()
        out = F.conv2d(x,weight,b,1,1)
        #frame = tf.cast(((frame - tf.reduce_min(frame)) / (tf.reduce_max(frame) - tf.reduce_min(frame))) * 255, tf.uint8)
        return out
    
    def forward(self, x):
        residual = x
        b,c,h,w = x.shape
        out = self.lrelu(self.input(x))
        out = self.lrelu(self.input2(out))
        out = self.res_feat1(out)
        x_detail = self.upscale_1(out)
        x_srbase = x_detail + F.interpolate(x, size=[h*self.scale, w*self.scale], mode="bilinear")
        
        x_fa = self.Laplacian(x_srbase)
        x_f = self.res_feat2(x_fa)
        res_in = x_f
        x_f = self.res_feat3(x_f)
        # mask
        frame_mask  = torch.sigmoid(self.res_feat4(res_in))
        x_frame = frame_mask* x_f + x_f
        x_frame = self.upscale_2(self.lrelu(self.reduce(x_frame)))
        #print(x_frame.shape, x_srbase.shape, x_fa.shape)
        x_sr = x_frame + x_srbase - x_fa
        return x_sr, x_srbase
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
parser.add_argument("--scale", type=int, default=8, help="scale size")
import time

#torch.cuda.set_device(2)
if __name__ == '__main__':
    opt = parser.parse_args()
    scale = opt.scale
    x = torch.rand(1,3,100,100)
    net = Net(scale).cuda()
    x = x.cuda()
    t0 = time.time()
    for i in range(30):
        out1,out12 = net(x)
    t = time.time() - t0
    print('average running time: ', t/30)
    count_parameters(net)
    print(out1.shape)
    #runing_time(net, x)

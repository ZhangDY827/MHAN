from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.SRCNN import Net as SRCNN
from models.VDSR import Net as VDSR
from models.LGCNet import Net as LGCNet
from models.EEGAN import Net as EEGAN
from models.model import Net as Mymodel
from models.DDBPN import DDBPN
from models.rcan import RCAN
from models.RDN import RDN
from models.san import Net as SAN
from data_x4 import get_training_set
import pdb
import socket
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument("--scale", type=int, default=4, help="super resolution upscale factor")
parser.add_argument("--n_colors", type=int, default=3, help="scale size")
parser.add_argument("--num_features", type=int, default=64, help="number of feature maps")
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=48, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=1500, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
parser.add_argument('--start_epoch', type=int, default=0, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='/titan_data1/lxy/dataset/Satellite_Imagery/AID_train')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--hr_train_dataset', type=str, default='AID_train_HR')
parser.add_argument('--model_type', type=str, default='EEGAN')
parser.add_argument('--patch_size', type=int, default=48, help='Size of cropped HR image')
parser.add_argument('--save_folder', default='weights', help='Location to save checkpoint models')
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument('--lr_train_dataset', type=str, default='/titan_data1/lxy/dataset/Satellite_Imagery/AID_train/AID_srcnn_x4')
opt = parser.parse_args()
gpus_list = range(opt.gpus)
cudnn.benchmark = True
print(opt)

def train(model, epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()
            target = target.cuda()
              
        optimizer.zero_grad()
        t0 = time.time()
        #prediction = model(input)
        out1,out2 = model(input)
        # print(prediction[0])
        # print(prediction.size)
        # print(target.size)  
        #loss = criterion(prediction, target)
        loss1 = criterion(out1, target)
        loss2 = criterion(out2, target)
        loss = 10 * loss1 + loss2
        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        if (iteration) % 50 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0)))
            #save_checkpoint(model, epoch)
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()
            target = target.cuda()
        
        prediction = model(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)

def save_checkpoint(model, epoch):
    model_folder = "checkpoint_EEGAN_x4/"
    model_out_path = model_folder + opt.model_type + "_x4" +"_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(state, model_out_path)
    

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.lr_train_dataset, opt.hr_train_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ', opt.model_type)

#model = SRCNN()
#model = VDSR()
#model = LGCNet()
#model = EEGAN(4)
model = Mymodel(3, 64, opt.scale)
#model = DDBPN(opt)
#model = RDN(opt.scale)
#model = RCAN(opt)
#model = SAN(opt)
#model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()#nn.MSELoss()#nn.L1Loss()

#################################
#################################

# pre_net = Mymodel_pre(3,64,4)
# pre_path = 'checkpoints_my_x4/model_my_x4_epoch_929.pth'
# checkpoint = torch.load(pre_path)
# pre_net.load_state_dict(checkpoint["model"].state_dict()) 

# pre_dict = pre_net.state_dict()
# model_dict = model.state_dict()

# pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}

# model_dict.update(pre_dict)
# model.load_state_dict(model_dict)

# print('Pre-train from:', pre_path)
################################
################################

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint["epoch"] + 1
        print(opt.start_epoch)
        model.load_state_dict(checkpoint["model"].state_dict())
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    train(model, epoch)

    # learning rate is decayed by a factor of 2 every 200 epochs
    if (epoch+1) % 500 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /=  10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
            
    if (epoch+1) % (opt.snapshots) == 0:
        save_checkpoint(model, epoch)

from __future__ import print_function
import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from srcnn import srcnn
from models.SRCNN import Net as SRCNN
from models.VDSR import Net as VDSR
#from models.DDBPN import DDBPN
from models.RDN import RDN
#from models.rcan import RCAN
#from models.model_my_2 import Net as Mymodel
#from data import get_eval_set
from functools import reduce
from PIL import Image, ImageOps
from torchvision.transforms import Compose, CenterCrop, ToTensor
from torchvision import transforms
from scipy.misc import imsave
import scipy.io as sio
import time
import cv2
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#torch.cuda.set_device(1)
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument("--scale", type=int, default=4, help="super resolution upscale factor")
parser.add_argument("--n_colors", type=int, default=3, help="scale size")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--self_ensemble', type=bool, default=False)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--input_dir', type=str, default='/titan_data1/lxy/dataset/Satellite_Imagery/RSSCN7/RSSCN7_LR/RSSCN7_LR_bicubic/X4')
parser.add_argument('--output', default='/titan_data1/zdy/Satellite/RSSCN7/Results', help='Location to save checkpoint models')
parser.add_argument('--test_dataset', type=str, default='RSSCN7_x2')
parser.add_argument('--model_type', type=str, default='srcnn_x2')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--model', default='weights/srcnn_x2_epoch_741.pth', help='sr pretrained base model')

opt = parser.parse_args()

gpus_list=range(opt.gpus)
print(opt)

cuda = False#opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
#test_set = get_eval_set(opt.input_dir, opt.upscale_factor)
#testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
#model = SRCNN()
#model = VDSR()
#model = DDBPN(opt)
#model = RCAN(opt)
model = RDN(opt.scale)
#model = Mymodel(3, 64, opt.scale)
# as we train the model on a single GPU, we are not supposed to use DataParallel which is for multi-GPUs model

# if cuda:
#     model = torch.nn.DataParallel(model, device_ids=gpus_list)
#     #print(model.keys())

# load model attention: first key is epoch, we should load the second key which is the entire model
checkpint = torch.load(opt.model,map_location='cpu')
pretrain_dict = checkpint['model']
# as you saved the entire model before, we should use '.state_dict()' to call the weights 
pretrain_dict = pretrain_dict.state_dict()
# this is the initial weights of your initial model
net_dict = model.state_dict()
# note that the structure of saved weights is a dictionary, and the keys between initial weights and pretrained weights are not match
# (keys in pretrained weights has a extral prefix '.module') Hence, we should delete that prefix to ensure they match each other.
pretrain_dict = {k : v for k, v in pretrain_dict.items() if  k in net_dict}
# we changed the keys of pretrained weights and we update it into the initial weights.
net_dict.update(pretrain_dict)
#print(net_dict)
# let the initial model load the updated weights
model.load_state_dict(net_dict)
print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda()

transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def eval(name):
    model.eval()


    t0 = time.time()
    filepath = './' + name
    input = Image.open(filepath).convert('RGB')
    #input = rescale_img(input, 4) #for SRCNN and VDSR
    input = transform(input).unsqueeze(0)
    _, name = os.path.split(filepath)
    if cuda:
        input = input.cuda()
    #print(input, input.shape)
    prediction = model(input)
    #print(prediction, prediction.shape)
    # if opt.residual:
    # prediction = prediction + bicubic

    t1 = time.time()
    print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
    save_img(prediction.cpu().data, name)

def save_img(img, img_name):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
    # save img
    #save_dir=os.path.join(opt.output, opt.test_dataset, opt.model_type)
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)
    save_fn = img_name.replace('x8','out_x8')
    print('saved in %s' % save_fn)
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])



##Eval Start!!!!
#name = '/titan_data1/lxy/dataset/Satellite_Imagery/RSSCN7/RSSCN7_LR/RSSCN7_LR_bicubic/X2/a001x2.png'
name1 = './img_half.jpg'
#name2 = './footballField_34x8.png'
#name = './real/Money200010.png'
eval(name1)
#eval(name2)

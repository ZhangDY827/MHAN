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
from models.DDBPN import DDBPN
from models.SRFBN import SRFBN
#from models.model_2 import Net as Mymodel
from models.model_x4 import Net as Mymodel_x4
from models.rcan import RCAN
#from models.model_my import Net as Mymodel
#from models.model_my_2 import Net as Mymodel
from models.RDN import RDN
from models.san import Net as SAN
from data_x8 import get_eval_set
from functools import reduce

from scipy.misc import imsave
import scipy.io as sio
import time
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#torch.cuda.set_device(1)
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument("--scale", type=int, default=8, help="super resolution upscale factor")
parser.add_argument("--num_features", type=int, default=64, help="number of feature maps")
parser.add_argument("--n_colors", type=int, default=3, help="scale size")
parser.add_argument('--testBatchSize', type=int, default=30, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--self_ensemble', type=bool, default=False)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--input_dir', type=str, default='/titan_data1/lxy/dataset/Satellite_Imagery/RSSCN7/RSSCN7_LR/RSSCN7_LR_bicubic/X2')
parser.add_argument('--output', default='/titan_data1/zdy/Satellite/WHU-RS19', help='Location to save checkpoint models')
parser.add_argument('--test_dataset', type=str, default='RSSCN7_x2')
parser.add_argument('--model_type', type=str, default='RCAN')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--model', default='weights/srcnn_x2_epoch_741.pth', help='sr pretrained base model')

opt = parser.parse_args()

gpus_list=range(opt.gpus)
print(opt)
cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
if opt.scale == 4 or opt.scale == 8:
    opt.input_dir = '/titan_data1/lxy/dataset/Satellite_Imagery/WHU-RS19/LR/X4'
if opt.scale == 2:
    opt.input_dir = '/titan_data1/lxy/dataset/Satellite_Imagery/WHU-RS19/LR/X2'

test_set = get_eval_set(opt.input_dir, opt.scale)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print('===> Building model')
#model = SRCNN()
#model = VDSR()
#model = DDBPN(opt)
#model = Mymodel(3, 64, opt.scale)
#model = Mymodel_x4(3, 64, opt.scale)
#model = SRFBN(opt.scale)
#model = RCAN(opt)
model = RDN(opt.scale)
#model = SAN(opt)
#model = Mymodel_ecan(opt)
# as we train the model on a single GPU, we are not supposed to use DataParallel which is for multi-GPUs model

# if cuda:
#     model = torch.nn.DataParallel(model, device_ids=gpus_list)
#     #print(model.keys())

# load model attention: first key is epoch, we should load the second key which is the entire model
checkpint = torch.load(opt.model)
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

def eval():
    model.eval()
    for batch in testing_data_loader:
        with torch.no_grad():
            input, name = Variable(batch[0]), batch[1]
        if cuda:
            input = input.cuda()
            # bicubic = bicubic.cuda(gpus_list[0])

        t0 = time.time()
        if opt.chop_forward:
            with torch.no_grad():
                prediction = chop_forward(input, model, opt.upscale_factor)
        else:
            if opt.self_ensemble:
                with torch.no_grad():
                    prediction = x8_forward(input, model)
            else:
                with torch.no_grad():
                    if opt.model_type == 'SRFBN':
                        prediction  = model(input)[-1]
                    else:
                        prediction = model(input)
                
        # if opt.residual:
            # prediction = prediction + bicubic

        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
        save_img(prediction.cpu().data, name[0])

def save_img(img, img_name):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
    # save img
    #save_dir=os.path.join(opt.output, opt.test_dataset, opt.model_type)
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)
    if opt.scale == 8:
        img_name = img_name.replace('x4', 'x8')
    save_fn = opt.output + '/' + img_name
    print('saved in %s' % save_fn)
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

def x8_forward(img, model, precision='single'):
    def _transform(v, op):
        if precision != 'single': v = v.float()

        v2np = v.data.cpu().numpy()
        if op == 'vflip':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'hflip':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 'transpose':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()
        
        ret = torch.Tensor(tfnp).cuda()

        if precision == 'half':
            ret = ret.half()
        elif precision == 'double':
            ret = ret.double()

        with torch.no_grad():
            ret = Variable(ret)

        return ret

    inputlist = [img]
    for tf in 'vflip', 'hflip', 'transpose':
        inputlist.extend([_transform(t, tf) for t in inputlist])

    outputlist = [model(aug) for aug in inputlist]
    for i in range(len(outputlist)):
        if i > 3:
            outputlist[i] = _transform(outputlist[i], 'transpose')
        if i % 4 > 1:
            outputlist[i] = _transform(outputlist[i], 'hflip')
        if (i % 4) % 2 == 1:
            outputlist[i] = _transform(outputlist[i], 'vflip')
    
    output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

    return output
    
def chop_forward(x, model, scale, shave=8, min_size=80000, nGPUs=opt.gpus):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
            if opt.self_ensemble:
                with torch.no_grad():
                    output_batch = x8_forward(input_batch, model)
            else:
                with torch.no_grad():
                    output_batch = model(input_batch)
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(x.data.new(b, c, h, w))

    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

##Eval Start!!!!
eval()

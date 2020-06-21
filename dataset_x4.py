import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    #y, _, _ = img.split()
    return img

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):

    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy,ix,iy + ip, ix + ip))
    img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))
    #img_bic = img_bic.crop((ty,tx,ty + tp, tx + tp))

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, info_patch

def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            info_aug['trans'] = True
            
    return img_in, img_tar, info_aug
    
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, lr_dir, patch_size, upscale_factor, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.HR ='/titan_data1/lxy/dataset/Satellite_Imagery/AID_train/AID_train_HR'
        self.LR ='/titan_data1/lxy/dataset/Satellite_Imagery/AID_train/AID_train_LR/x4'
        #self.LR ='/titan_data1/lxy/dataset/Satellite_Imagery/AID_train/AID_srcnn_x4'
# viaduct_420x2.png
# viaduct_420.jpg
    def __getitem__(self, index):
        target = load_img(self.image_filenames[index])
        # print(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])

        input = load_img(os.path.join(self.LR, file))
        # print(self.LR,os.path.splitext(file)[0],'x2x2.png')
        # input = target.resize((int(target.size[0]/self.upscale_factor),int(target.size[1]/self.upscale_factor)), Image.BICUBIC)
        # input = rescale_img(input, self.upscale_factor)      
        # input = input.resize((int(target.size[0]),int(target.size[1])), Image.BICUBIC) 
        # bicubic = input

        
        # input, target, bicubic, _ = get_patch(input,target,bicubic,self.patch_size, self.upscale_factor)
        input, target, _ = get_patch(input,target,self.patch_size, self.upscale_factor)
        if self.data_augmentation:
            # input, target, bicubic, _ = augment(input, target, bicubic)
            input, target, _ = augment(input, target)

        if self.transform:
            input = self.transform(input)
            # bicubic = self.transform(bicubic)
            target = self.transform(target)
                
        return input, target

    def __len__(self):
        return len(self.image_filenames)

class DatasetFromFolderEval(data.Dataset):
    def __init__(self, lr_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.image_filenames = [join(lr_dir, x) for x in listdir(lr_dir) if is_image_file(x)]
        # self.upscale_factor = upscale_factor
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        _, file = os.path.split(self.image_filenames[index])
        #input = rescale_img(input, 4)
        #input = input.resize((int(target.size[0]),int(target.size[1])), Image.BICUBIC) 

        # bicubic = input
        
        if self.transform:
            input = self.transform(input)
            # bicubic = self.transform(bicubic)
            
        return input, file
      
    def __len__(self):
        return len(self.image_filenames)

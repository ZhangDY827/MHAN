from os.path import join
from torchvision.transforms import Compose, ToTensor
from dataset_x4 import DatasetFromFolderEval, DatasetFromFolder

def transform():
    return Compose([
        ToTensor(),
    ])

def get_training_set(data_dir, dataset, hr, upscale_factor, patch_size, data_augmentation):
    hr_dir = join(data_dir, hr)
    lr_dir = join(data_dir, dataset)
    return DatasetFromFolder(hr_dir, lr_dir, patch_size, upscale_factor, data_augmentation,
                             transform=transform())

def get_eval_set(lr_dir, upscale_factor):
    return DatasetFromFolderEval(lr_dir,
                             transform=transform())


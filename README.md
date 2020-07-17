# Remote Sensing Image Super-Resolution via Mixed High-Order Attention Network (MHAN) (Accept by TGRS 2020)
The is the pytorch code for paper "Remote Sensing Image Super-Resolution via Mixed High-Order Attention Network" [MHAN](https://doi.org/10.1109/TGRS.2020.3009918).
The Test30 dataset used in the paper can be found in this repository，referring to the folder './Test30'.
Some other general image and remote sensing SR based models also provided in folder './models'.

# Requirements

- Python 3.6.4
- Pytorch 1.3.1(GPU)
- OpenCV
- NVIDIA-SMI 430.64       
- Driver Version: 430.64       
- CUDA Version: 10.1  
# Dataset
We use [AID](https://arxiv.org/abs/1608.05167v1) as the training dataset, which is a
collection of remote sensing images depicting 30 land-use
classes, including airport, farmland, beach, desert, etc.

We conducted experiments on two satellite image datasets, 
namely, [WHURS19](http://www.escience.cn/people/yangwen/WHU-RS19.html) 
and [RSSCN7](https://hyper.ai/datasets/5440).

# Usage
Use the following command to train the model. 
```
$ python main_x4.py
```
Use the following commandss to generate the SR images with respect to RSSCN7 and WHURS19 datasets. 
```
$ python eval_RSSCN7.py
$ python eval_WHURS19.py
```
When the SR images are generated in the folder, use Evaluate_PSNR_SSIM.m file to comptute the PSNR and SSIM.


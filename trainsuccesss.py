import argparse
from ast import arg
import logging
import sys
from pathlib import Path
import os
from tokenize import String
os.environ["CUDA_VISIBLE_DEVICES"]="3" 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import int32, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataLoad import *
from utils.dice_score import *
from test import evaluate
from model import *
from utils.loss_function import *
import numpy as np
import pandas as pd
from train.train_part import train_net
from flashUnet import FlashUnet
from transUnet import VisionTransformer
# from unetsuccessful import *
# from transUnet import TransUnet

dir_img = r'noduleData/2d/dataset(all)/train/images'
dir_mask = r'noduleData/2d/dataset(all)/train/masks'
dir_img_val = r'noduleData/2d/dataset(all)/test/images'
dir_mask_val = r'noduleData/2d/dataset(all)/test/images'
dir_checkpoint = r'Pytorch-UNet-2.0/Pytorch-UNet-2.0/experiment'


# 获取参数
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=15, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--isStep', default=False, help='')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    return parser.parse_args()

# 程序主体
if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print(torch.cuda.is_available())   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    # net = Unet(1,2,isStep=args.isStep)
    # net  = VUnet(1,2)
    # net = FlashUnet()
    # net = AttU_Net()
    # net = VisionTransformer()
    # net = ResNet34UnetPlus(num_channels=1,num_class=2)
    net  = FLASHUnet(1,2)
    # net = ResNet([3,4,6,3],1,2)
    # net = SegNet(1,2)
    # net = ResNet34UnetPlusFLASH(1,2)
    # net = FCN32s(in_class=1,n_class=2)
    # net.load_from(weights=np.load('R50+ViT-B_16.npz'))
    print(net)

    # net  = FLASHUnet(1,2)
    # 获取网络的参数量
    total = sum([param.nelement() for param in net.parameters()])
    print("Numberof parameter: % .2fM" % (total / 1024**2))
    # net = DFDUnet(1,2,isStep=args.isStep)
    logging.info(f'Network:\n'
                 f'\t1 input channels\n'
                 f'\t2 output channels (classes)\n'
                 f'\t"Transposed conv" upscaling')
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
    net.to(device=device)

    # 加载数据
    test_dataset=ImgDataLoader(dir_img_val,dir_mask_val)
    dataset=ImgDataLoader(dir_img,dir_mask)
    l = dataset.__len__()
    n_train = int(0.8*l)
    n_val = l-n_train
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    # 3. Create data loaders pin_memory设置为True表示不能够与虚拟内存进行数据交换
    loader_args = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args,drop_last=False)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, **loader_args)

    # 保存路径
    save_path = r'model_data/baseline/flashunet-best/2'
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=1.0,
                  save_path=save_path,
                  amp=args.amp,
                  n_train=n_train,
                  n_val=n_val)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)

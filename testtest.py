from test import evaluate, evaluate2D
from model import *
from utils.dice_score import multiclass_dice_coeff
from utils.loss_function import *
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import optim
import pandas as pd
import wandb
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse
import logging
import sys
from dataLoad import *
from pathlib import Path
import os
# from successful.models.dfSuccess import DFUNet2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

dir_img_val = r'noduleData/2d/dataset(all)/test/images'
dir_mask_val = r'noduleData/2d/dataset(all)/test/masks'

dir_checkpoint = r'Pytorch-UNet-2.0/Pytorch-UNet-2.0/experiment'


val_set = ImgDataLoader(dir_img_val, dir_mask_val,isName=True)


val_loader = DataLoader(val_set, batch_size=1, shuffle=False, drop_last=True)


# 程序主体
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    print(torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # net = DFDUnet2(1, 2)
    totalPath = 'model_data/baseline/attention-unet/model/'
    netPath = totalPath+'net.h5'
    modelPath = totalPath+'youwillWin.pth'
    net = torch.load(netPath)
    print(net)
    
    # net =DFDUnet(1,2)
    # net = Unet(1, 2)
    queryset = pd.read_csv('noduleData/2d/dataset(all)/allofData.csv')
    net.load_state_dict(torch.load(modelPath, map_location=device))
    net=net.to(device=device)
    # net.load_state_dict(torch.load('Pytorch-UNet-2.0/Pytorch-UNet-2.0/experiment/unetSuccess!/youwillwin.pth', map_location=device))
    # net=net.to(device=device)
    print(evaluate2D(net, val_loader, device=device,queryset=queryset, isStep=False))
    # (Initialize logging)
    # experiment = wandb.init(project='Result_Win', resume='allow', anonymous='must')
    # experiment.config.update(dict(epochs=1, batch_size=1, learning_rate=1,
    #                               val_percent=1, save_checkpoint=1, img_scale=1,
    #                               amp=1))
    # global_step = 0

    # for image,true_masks in val_loader:
    #     # print(val_loader.__len__())
    #     # print(image.shape)
    #     # print(mask.shape)
    #     print(global_step)
    #     with torch.no_grad():
    #         # predict the mask
    #         mask_pred = net(image.to(device=device))

    #         mask_preds = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()

    #         true_maskss = F.one_hot(true_masks, 2).permute(0, 3, 1, 2).float()

    #         # compute the Dice score, ignoring background
    #         dice_score = multiclass_dice_coeff(mask_preds[:, 1, ...].unsqueeze(0), true_maskss[:, 1, ...].unsqueeze(0).to(device=device), reduce_batch_first=False)

    #         experiment.log({
    #         'validation Dice': dice_score,
    #         'images': wandb.Image(image[0].cpu()),
    #         'masks': {
    #             'true': wandb.Image(true_masks[0].float().cpu()),
    #             'pred': wandb.Image(torch.softmax(mask_pred, dim=1)[0][1].float().cpu()),
    #         },
    #         'index': global_step,
    #         'epoch': 1
    #         })
    #     global_step = global_step+1

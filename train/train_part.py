import argparse
import logging
import sys
from pathlib import Path
import os
from tokenize import String
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import int32, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils.dice_score import *
from test import evaluate2D,evaluate
from model import *
from utils.loss_function import *
import numpy as np
import pandas as pd

def train_net(net,
              device,
              train_loader,
              val_loader,
              save_path,
              epochs: int = 1,
              batch_size: int = 1,
              learning_rate: float = 0.0001,
              val_percent: float = 1.0,
              save_checkpoint: bool = True,
              img_scale: float = 1.0,
              amp: bool = False,
              isStep: bool = False,
              n_train=None,
              n_val=None):


    # 定义数组
    train_dice_array = []
    test_dice_array = []
    train_loss = []

    # (Initialize logging)
    experiment = wandb.init(project='Flash-Net-Win!', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    diceloss=DiceLoss()
    # 保存最好的模型
    bestValue = 0
    data_path=save_path+'/data'
    model_path = save_path+'/model'
    if not os.path.exists(save_path): os.mkdir(save_path)
    if not os.path.exists(data_path): os.mkdir(data_path)
    if not os.path.exists(model_path): os.mkdir(model_path)
    torch.save(net,model_path+'/net.h5')
    bestEpoch = 0
    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for image,mask in train_loader:
                images = image
                true_masks = mask
                assert images.shape[1] == 1, \
                    f'Network has been defined with 1 input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                with torch.cuda.amp.autocast(enabled=amp):
                    # 结果和过程
                    if isStep:
                        masks_pred,steps_pred1,steps_pred2,steps_pred3,steps_pred4,steps_pred5 = net(images)
                    else:
                       masks_pred =  net(images)   
                    loss = criterion(masks_pred, true_masks) \
                           + diceloss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, 2).permute(0, 3, 1, 2).float())
                train_loss.append(loss.item())
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
            histograms = {}
            for tag, value in net.named_parameters():
                tag = tag.replace('/', '.')
                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                # histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())            
            # 验证集准确度
            val_score = evaluate(net, val_loader, device,isStep=isStep)
            # val_score = val_score['dice']
            test_dice_array.append(val_score.item())
            # 训练集准确度
            train_score = evaluate(net, train_loader, device,isStep=isStep)
            # train_score = train_score['dice']
            train_dice_array.append(train_score.item())    
            # 保存最好的模型
            if(val_score>bestValue):
                bestValue =val_score
                torch.save(net.state_dict(), model_path+ '/youwillWin.pth')
                bestEpoch = epoch+1
                print(' Best Save!')
            print('Best performance at Epoch: {} | {}'.format(bestEpoch,bestValue))
            # 调整训练参数
            scheduler.step(val_score)
            # 如果需要打印过程
            if isStep:            
                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'train Dice':train_score,
                    'validation Dice': val_score,
                    'images': wandb.Image(images[0].cpu()),
                    'step_images1': wandb.Image(steps_pred1[0][0].cpu()),
                    'step_images2': wandb.Image(steps_pred2[0][0].cpu()),
                    'step_images3': wandb.Image(steps_pred3[0][0].cpu()),
                    'step_images4': wandb.Image(steps_pred4[0][0].cpu()),
                    'step_images5': wandb.Image(steps_pred5[0][0].cpu()),
                    'masks': {
                        'true': wandb.Image(true_masks[0].float().cpu()),
                        'pred': wandb.Image((torch.softmax(masks_pred, dim=1)[1]>=0.5).float().cpu())
                    },
                    'step': global_step,
                    'epoch': epoch,
                    **histograms
                })
            else:
                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'train Dice':train_score,
                    'validation Dice': val_score,
                    'images': wandb.Image(images[0].cpu()),
                    'masks': {
                        'true': wandb.Image(true_masks[0].float().cpu()),
                        'pred': wandb.Image((torch.softmax(masks_pred, dim=1)[0][1]>=0.5).float().cpu())
                    },
                    'step': global_step,
                    'epoch': epoch,
                    **histograms
                })
            # 保存训练结果，train_loss,test_dice,train_dice
            datas = pd.DataFrame(data=
            {
            'test_dice':test_dice_array,
            'train_dice':train_dice_array
            })
            datas.to_csv(data_path+'/train_data_dice.csv')
            print("训练数据保存成功")

            # 保存训练结果，train_loss,test_dice,train_dice
            datas = pd.DataFrame(data=
            {
            'train_loss':train_loss,
            })
            datas.to_csv(data_path+'/train_data.csv')
            print("训练数据保存成功")

        
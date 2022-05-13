import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.dice_score import multiclass_dice_coeff
from utils import metrics


def evaluate(net, dataloader, device,isStep=False):
    net.eval()
    num_val_batches = len(dataloader)
    # print(num_val_batches)
    dice_score = 0
    # iterate over the validation set
    for images,mask in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = images,mask
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
        # return  mask_true[:, 1, ...].unsqueeze(0)
        

        with torch.no_grad():
            # predict the mask
            if isStep:
                mask_pred,_,_,_,_,_ = net(image)
            else:
                mask_pred = net(image)

            mask_pred = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()

            # compute the Dice score, ignoring background
            items=multiclass_dice_coeff(mask_pred[:, 1, ...].unsqueeze(0), mask_true[:, 1, ...].unsqueeze(0), reduce_batch_first=False)
            dice_score += items

    net.train()
    return dice_score / num_val_batches


def evaluate2D(net, dataloader, device,n_labels=2,queryset=None,isStep=False):
    net.eval()
    num_val_batches = len(dataloader)
    val_dice = metrics.DiceAverage(n_labels)
    val_sensity=metrics.SensitivityAverage(n_labels)
    val_ppv=metrics.PPVAverage(n_labels)
    val_asd=metrics.ASDAverage(n_labels)
    value=[]
    stvalue=[]
    ppvalue=[]
    asdvalue=[]
    times = []
    # print(num_val_batches)
    dice_score = 0
    # iterate over the validation set
    if queryset is None:
        for images,mask in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = images,mask
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
            with torch.no_grad():
                # predict the mask
                if isStep:
                    mask_pred,_,_,_,_,_ = net(image)
                else:
                    mask_pred = net(image)
                # print(mask_pred)
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
                # print(mask_pred)
                # mask_pred = mask_pred.unsqueeze(2)
                # mask_true = mask_true.unsqueeze(2)
                val_dice.update(mask_pred, mask_true)
                val_sensity.update(mask_pred, mask_true)
                val_ppv.update(mask_pred, mask_true)
                # val_asd.update(output, target)
                value.append(val_dice.value[1])
                stvalue.append(val_sensity.value[1])
                ppvalue.append(val_ppv.value[1])
        net.train()
        return {'dice':val_dice.avg[1],'std-dice':np.std(value),
                'sensity':val_sensity.avg[1],'std-sensity':np.std(stvalue),
                'ppv':val_ppv.avg[1],'std-ppv':np.std(ppvalue)
        }
    else:
        for images,mask,imgpath in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = images,mask
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
            with torch.no_grad():
                # predict the mask
                if isStep:
                    mask_pred,_,_,_,_,_ = net(image)
                else:
                    start = time.clock()
                    mask_pred = net(image)
                    end = time.clock()
                    times.append(end-start)
                # print(mask_pred)
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
                # print(mask_pred)
                # mask_pred = mask_pred.unsqueeze(2)
                # mask_true = mask_true.unsqueeze(2)
                val_dice.update(mask_pred, mask_true)
                val_sensity.update(mask_pred, mask_true)
                val_ppv.update(mask_pred, mask_true)
                xyz = getThePositionOfXYZ(queryset=queryset,path=imgpath[0])
                val_asd.update(mask_pred, mask_true,xyz)
                asdvalue.append(val_asd.value[1])
                value.append(val_dice.value[1])
                stvalue.append(val_sensity.value[1])
                ppvalue.append(val_ppv.value[1])
        net.train()
        return {'dice':val_dice.avg[1],'std-dice':np.std(value),
                'sensity':val_sensity.avg[1],'std-sensity':np.std(stvalue),
                'ppv':val_ppv.avg[1],'std-ppv':np.std(ppvalue),
                'asd':val_asd.avg[1],'std-asd':np.std(asdvalue),
                "time cost":np.mean(times)
        }


def evaluate2DASD(net, dataloader, device,queryset,n_labels=2,isStep=False):
    net.eval()
    num_val_batches = len(dataloader)
    val_asd=metrics.ASDAverage(n_labels)
    value=[]
    # print(num_val_batches)
    dice_score = 0
    # iterate over the validation set
    for images,mask,imgpath in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # print(imgpath[0])
        image, mask_true = images,mask
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, 2).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            # predict the mask
            if isStep:
                mask_pred,_,_,_,_,_ = net(image)
            else:
                mask_pred = net(image)
            # print(mask_pred)
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), 2).permute(0, 3, 1, 2).float()
            xyz = getThePositionOfXYZ(queryset=queryset,path=imgpath[0])
            val_asd.update(mask_pred, mask_true,xyz)
            value.append(val_asd.value[1])
    net.train()
    return {'asd':val_asd.avg[1],'std-asd':np.std(value),
    }
# 获取数据索引号
def getTheindex(path):
    length = len(path)
    return int(path[:length-4]) 

# 返回x,y,z坐标
def getThePositionOfXYZ(queryset,path):
    iid = getTheindex(path)
    x = queryset['x'][iid]
    y = queryset['y'][iid]
    z = queryset['z'][iid]
    return [x,y,z]


           



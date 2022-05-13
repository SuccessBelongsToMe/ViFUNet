import sys
from torch.functional import Tensor
sys.path.append('D:\\anaconda\\Lib\\site-packages')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
import pandas as pd
import joblib
from collections import OrderedDict
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot
from torch.utils.data import DataLoader
from vit import *

def dataConvert(path):
    img=Image.open(path)
    transform=Compose([Resize((224,224)),ToTensor()])
    x=transform(img)
    x=x.unsqueeze(0)
    return x

modal_path=r'vit.pth'
# 定义相关参数
batch_size=1
# fig=plt.figure()
# plt.imshow(img)
# plt.show()
# 描点图片
origin_input=dataConvert(r'vit\data\origin\1.jpg')
# 正样本图片
right_input=dataConvert(r'vit\data\transform\1(1).jpg')
# 负样本图片
wrong_input1=dataConvert(r'vit\data\origin\2.jpg')
wrong_input2=dataConvert(r'vit\data\origin\3.jpeg')
wrong_input3=dataConvert(r'vit\data\origin\3.jpeg')


# o=torch.Tensor(1,768)
# r=torch.Tensor(1,768)
# F.pairwise_distance(o,r)
# 使用GPU或者CPU
if torch.cuda.is_available():
    device = torch.device("cuda")  
else:
    device = torch.device("cpu")  
# device = torch.device("cpu") 
print(device.type)
# 建立模型

vitModule=ViT()
# deformUnetModule=testNet(3,2)
# simpleModule=myNet(3,2)
vitModule.to(device=device)


m=torch.tensor(5)
# print(m)
lossFunc=TripletLossFunc(m)
lossFunc.to(device=device)
# loss=lossFunc(o,r,w)
# print(loss.backward())
# input=input.to(device)
# pre=pre.to(device=device)
# 设置训练参数
epoch=300
# lossFunction1=nn.CrossEntropyLoss()
# lossFunction1.to(device)
optimizer=torch.optim.SGD(vitModule.parameters(),lr=0.01)
# 绘制曲线
losslist=[]

# 开始训练
for i in range(epoch):   
    # for batch in train_data:
    # input=batch["image"].to(device=device)                                                                                                                                                                  
    # input=input.requires_grad_()#生成变量 
    o=vitModule(origin_input.to(device=device))
    r=vitModule(right_input.to(device=device))
    w1=vitModule(wrong_input1.to(device=device))
    w2=vitModule(wrong_input2.to(device=device))
    w3=vitModule(wrong_input3.to(device=device))
    w=torch.stack([w1,w2,w3],dim=0).to(device=device) 
    #卷
    # output=ViT(input)
    # output=output.requires_grad_()#生成变量 
    # 预测值
    # pre=batch["pre"].to(device=device)    
    # 计算损失函数
    loss=lossFunc(o,r,w)     
    # 输出损失值
    print("loss:  "+str(loss.item()))
    losslist.append(loss.item())
    # 梯度清零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 权值更新
    optimizer.step()
torch.save(vitModule.state_dict(),modal_path)
# 绘制损失函数图像
y=losslist
x=range(0,y.__len__())
print(y.__len__())
pyplot.subplot(2, 1, 2)
pyplot.plot(x, y, '.-')
pyplot.xlabel('Test loss vs. epoches')
pyplot.ylabel('Test loss')
pyplot.show()
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np

class LNLoader(Dataset):

    def __init__(self,alldata,maskdata):
        self.data=alldata
        self.mask=maskdata
        self.len=alldata.shape[0]
    def __getitem__(self,item):    
        input=torch.Tensor(self.data[item]).unsqueeze(0).float()
        # 训练图片
        m=self.mask[item]
        mask_data=torch.Tensor(m).long()
        return {"image":input,"pre":mask_data}

    def __len__(self):
        return self.len

     

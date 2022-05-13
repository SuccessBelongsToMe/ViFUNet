from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np

class ImgDataLoader(Dataset):
    def __init__(self,imag_path,mask_path,isName=None):
        self.imag_path=imag_path
        self.mask_path=mask_path
        self.imag_path_list=os.listdir(imag_path)
        self.imag_path_list.sort()
        self.mask_path_list=os.listdir(mask_path)
        self.mask_path_list.sort()
        self.isName=isName
    def __getitem__(self,item):
        imag_name=self.imag_path_list[item]
        imag_item_path=os.path.join(self.imag_path,imag_name)
        img=np.load(imag_item_path)
        # imag=Image.open(imag_item_path)
        # img=np.array(imag)
        img_data=torch.from_numpy(img).float()
        input=img_data.unsqueeze(0)
        # 训练图片
        mask_name=self.mask_path_list[item]
        mask_item_path=os.path.join(self.mask_path,mask_name)
        m=np.load(mask_item_path)
        # mask=Image.open(mask_item_path)
        # m=np.array(mask)
        mask_data=torch.from_numpy(m).long()

        # mask=torch.permute(mask_data,[1,0])
        if self.isName:
            return input,mask_data,imag_name
        return input,mask_data

    def __len__(self):
        return len(self.imag_path_list)

     

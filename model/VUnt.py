from turtle import forward
import torch.nn as nn
import torch
from torch import Tensor, autograd, dropout
from torchvision.ops.deform_conv import DeformConv2d
from .vit import ViT

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, input):
        return self.conv(input)



 
 
class VUnet(nn.Module):
    def __init__(self,in_ch,out_ch,isStep=False):
        super(VUnet, self).__init__() 
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1) 
        self.isStep = isStep

        self.vit = ViT(in_channels=256,patch_size=1,emb_size=256,img_size=8,depth=12,n_classes=2)
    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        # c2: 得到32*32,128通道
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        # c3: 得到16*16,256通道
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        #cc: 得到8*8,512通道
        cc=self.conv4(self.vit(p3))
        # 得到8*8没有改变图像尺寸，512通道
        # c4=self.conv4(p3)
        # p4=self.pool4(c4)
        # c5=self.conv5(p4)
        # up_6= self.up6(c5)
        # merge6 = torch.cat([up_6, c4], dim=1)
        # c6=self.conv6(merge6)
        # up_7=self.up7(c6)
        # merge7 = torch.cat([up_7, c3], dim=1)
        # c7=self.conv7(merge7)

        # up_7: 得到16*16，256通道
        up_7=self.up7(cc)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        #up_8: 得到32*32，128通道
        up_8=self.up8(c7)
        merge8=torch.cat([up_8,c2],dim=1)
        c8=self.conv8(merge8)
        #up_9: 得到64*64，64通道
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)     
        c10=self.conv10(c9)
        out = nn.Softmax(dim=1)(c10)
        return out
    
if __name__ == '__main__':
    model = VUnet(1, 2)
    total = sum([param.nelement() for param in model.parameters()])
    print("Numberof parameter: % .2fM" % (total / 1024**2))
    input = torch.randn(size=(2, 1,64, 64))
    out = model(input)
    print(out.shape)
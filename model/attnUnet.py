from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi

class AttU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=2):
        super(AttU_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)   # [64,512,512]

        x2 = self.Maxpool(x1) # [64,256,256]
        x2 = self.Conv2(x2)  # [128,256,256]

        x3 = self.Maxpool(x2) # [128,128,128]
        x3 = self.Conv3(x3)  # [256,128,128]

        x4 = self.Maxpool(x3) # [256,64,64]
        x4 = self.Conv4(x4)  # [512,64,64]

        x5 = self.Maxpool(x4) # [512,32,32]
        x5 = self.Conv5(x5)  # [1024,32,32]

        # decoding + concat path
        d5 = self.Up5(x5)    # [512,64,64]
        # print(d5.shape)    # [512,8,8]
        # d5 = self.flash_vit1(d5)
        # x4 = self.flash_vit1(x4)
        x4 = self.Att5(g=d5, x=x4)  # [512,64,64],[512,64,64]
        d5 = torch.cat((x4, d5), dim=1)  # [1024,64,64]
        d5 = self.Up_conv5(d5)  # [512,64,64]

        d4 = self.Up4(d5)    # [256,128,128]
        # d4 = self.flash_vit2(d4)
        # x3 = self.flash_vit2(x3)
        x3 = self.Att4(g=d4, x=x3)  # [256,128,128],[256,128,128]
        d4 = torch.cat((x3, d4), dim=1)  # [512,128,128]
        d4 = self.Up_conv4(d4)  # [256,128,128]

        d3 = self.Up3(d4)    # [128,256,256]
        x2 = self.Att3(g=d3, x=x2)  # [128,256,256],[128,256,256]
        # d3 = self.flash_vit3(d3)
        # x2 = self.flash_vit3(x2)
        d3 = torch.cat((x2, d3), dim=1)  # [256,256,256]
        d3 = self.Up_conv3(d3)  # [128,256,256]

        d2 = self.Up2(d3)    # [64,512,512]
        # d2 = self.flash_vit4(d2)
        # x1 = self.flash_vit4(x1)
        x1 = self.Att2(g=d2, x=x1)  # [64,512,512],[64,512,512]
        d2 = torch.cat((x1, d2), dim=1)  # [128,512,512]
        d2 = self.Up_conv2(d2)  # [64,512,512]

        d1 = self.Conv_1x1(d2)  # [2,512,512]
        d1 = self.sigmoid(d1)   # [2,512,512]

        return d1

if __name__ == '__main__':
    model = AttU_Net()
    total = sum([param.nelement() for param in model.parameters()])
    print("Numberof parameter: % .2fM" % (total / 1024**2))
    input = torch.randn(size=(2, 1,64, 64))
    out = model(input)
    print(out.shape)

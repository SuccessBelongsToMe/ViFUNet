import torch.nn as nn
import torch
from torch import Tensor, autograd
from torchvision.ops.deform_conv import DeformConv2d

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch,mid_ch=None):
        super(DoubleConv, self).__init__()
        if (mid_ch == None):
            mid_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ) 
    def forward(self, input):
        return self.conv(input)

class DoubleDFConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleDFConv, self).__init__()
        self.conv = nn.Sequential(
            DeformConv2dPack(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            DeformConv2dPack(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, input):
        return self.conv(input)

class DeformConv2dPack(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=True, modulation=True):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2dPack, self).__init__()
        self.offset=nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        self.defconv=DeformConv2d(inc,outc,3,1,1)
      
    def forward(self, x):
        offset=self.offset(x)
        return self.defconv(x,offset)

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

 
 
class DFDUnet2(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DFDUnet2, self).__init__() 
        self.dfconv1 = DoubleDFConv(in_ch, 32)
        self.dfconv2 = DoubleConv(32, 64)
        self.dfconv3 = DoubleConv(64, 128)
        self.dfconv4 = DoubleConv(128, 256)

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
        self.dfconv6=DoubleConv(1024+256, 512)
        self.dfconv7=DoubleConv(512+128, 256,256)
        self.dfconv8=DoubleConv(256+64, 128,128)
        self.dfconv9=DoubleConv(128+32, 64)
        self.dfconv10=nn.Conv2d(64,out_ch, 1) 
        self.conv10 = nn.Conv2d(64,out_ch, 1) 
    def forward(self,x):
        dc1 = self.dfconv1(x)
        dcc1 = self.pool1(dc1)
        dc2 = self.dfconv2(dcc1)
        dcc2 = self.pool1(dc2)
        dc3=self.dfconv3(dcc2)
        dcc3 = self.pool1(dc3)
        dc4=self.dfconv4(dcc3)
        with torch.no_grad():
            c1=self.conv1(x)
            p1=self.pool1(c1)
            c2=self.conv2(p1)
            p2=self.pool2(c2)
            c3=self.conv3(p2)
            p3=self.pool3(c3)
            c4=self.conv4(p3)
            p4=self.pool4(c4)
            c5=self.conv5(p4)
            up_6= self.up6(c5)
        merge6 = torch.cat([dc4, up_6, c4], dim=1)
        c6=self.dfconv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([dc3, up_7, c3], dim=1)
        c7=self.dfconv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([dc2, up_8, c2], dim=1)
        c8=self.dfconv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([dc1,up_9,c1],dim=1)
        c9=self.dfconv9(merge9)
        c10=self.dfconv10(c9)
        out = nn.Softmax(dim=1)(c10)
        return out


      
if __name__ == '__main__':
    model = DFDUnet2(1, 2)
    total = sum([param.nelement() for param in model.parameters()])
    print("Numberof parameter: % .2fM" % (total / 1024**2))
    input = torch.randn(size=(2, 1,64, 64))
    out = model(input)
    print(out.shape)
from turtle import forward
import torch.nn as nn
import torch
from torch import Tensor, autograd
from torchvision.ops.deform_conv import DeformConv2d
from modulated_deform_conv import *

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch,mid_ch=None):
        super(DoubleConv, self).__init__()
        if mid_ch == None:
            mid_ch=out_ch
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
            DeformConv(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            DeformConv(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, input):
        return self.conv(input)
# 可变形卷积
class DeformConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.offset1=nn.Conv2d(in_channels,2*3*3,kernel_size=3,padding=1)
        self.DConv1=DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1)

# 前向传播，上一层的输出是下一层的输入
    def forward(self, x):
        # 获取偏置项1
        offset1=self.offset1(x)
        # 进行一次可变形卷积
        d_x1=self.DConv1(x,offset1)
        return d_x1

 
 
class WINUnet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(WINUnet, self).__init__() 
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
    def forward(self,x):
    #   with torch.no_grad():
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
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        return merge8,merge9

class DeformConv2dSuccess(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=False, modulation=True):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2dSuccess, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.dfconv = ModulatedDeformConv2d(inc,outc,3,1,1)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # offset = torch.zeros(x.size(0),18,x.size(2),x.size(3)).to(device=torch.device('cuda'))
        offset = self.p_conv(x)
        # print(offset.shape)
        # np.save("offser2d",offset.detach().cpu().numpy())
        # np.save("data2d",x.detach().cpu().numpy())
        # print(offset)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
        # dtype = offset.data.type()
        # ks = self.kernel_size
        # N = offset.size(1) // 2
        # if self.padding:
        #     x = self.zero_padding(x)
        # # (b, 2N, h, w)
        # p = self._get_p(offset, dtype)
        # # (b, h, w, 2N)
        # p = p.contiguous().permute(0, 2, 3, 1)
        # q_lt = p.detach().floor()
        # q_rb = q_lt + 1
        # q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        # q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        # # clip p
        # p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
        # # bilinear kernel (b, h, w, N)
        # g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        # g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        # g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        # g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        # # (b, c, h, w, N)
        # x_q_lt = self._get_x_q(x, q_lt, N)
        # x_q_rb = self._get_x_q(x, q_rb, N)
        # x_q_lb = self._get_x_q(x, q_lb, N)
        # x_q_rt = self._get_x_q(x, q_rt, N)
        # # (b, c, h, w, N)
        # x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
        #            g_rb.unsqueeze(dim=1) * x_q_rb + \
        #            g_lb.unsqueeze(dim=1) * x_q_lb + \
        #            g_rt.unsqueeze(dim=1) * x_q_rt
        # # modulation
        # if self.modulation:
        #     m = m.contiguous().permute(0, 2, 3, 1)
        #     m = m.unsqueeze(dim=1)
        #     m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
        #     x_offset *= m
        # x_offset = self._reshape_x_offset(x_offset, ks)
        # out = self.conv(x_offset)
        out = self.dfconv(x,offset,m)
        return out

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

class WinNet(nn.Module):
  def __init__(self,in_ch,out_ch,model=None):
      super(WinNet, self).__init__() 
      self.WINUnet = WINUnet(in_ch,out_ch)
      if(model != None):
        self.WINUnet.load_state_dict(model)
      self.dfconv = nn.Sequential(
        DeformConv2dSuccess(128,64),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        DeformConv2dSuccess(64,64),
        nn.BatchNorm2d(64),
        nn.ReLU()
      )
      self.conv = nn.Sequential(
        nn.Conv2d(128,64,3,1,1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        DeformConv2dSuccess(64,64,3,1,1),
        nn.BatchNorm2d(64),
        nn.ReLU()
      )

      self.dfconv2 = nn.Sequential(
        DeformConv2dSuccess(256,128),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        DeformConv2dSuccess(128,128),
        nn.BatchNorm2d(128),
        nn.ReLU()
      )
      self.conv2 = nn.Sequential(
        nn.Conv2d(256,128,3,1,1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        DeformConv2dSuccess(128,128,3,1,1),
        nn.BatchNorm2d(128),
        nn.ReLU()
      )
      self.up = nn.ConvTranspose2d(256, 128, 2, stride=2)

      self.conv9 = DoubleConv(128, 64)
      self.conv10 = nn.Conv2d(64,out_ch, 1) 


  def forward(self,x):
    lowFeature,heightFeature= self.WINUnet(x)
    # print(lowFeature.shape)
    # print(heightFeature.shape)
    h1=self.dfconv(heightFeature)
    h2=self.conv(heightFeature)
    # h11 =self.dfconv2(lowFeature)
    # h12=self.conv2(lowFeature)
    # print(h11.shape)
    # print(h12.shape)
    # hall = torch.cat([h11,h12],dim=1)
    # hbig =self.up(hall)
    # print(hbig.shape)
    # hAll = torch.cat([h1,h2,hbig],dim=1)
    hAll = torch.cat([h1,h2],dim=1)
    # print(hAll.shape)
    hs = self.conv9(hAll)
    # print(hs.shape)
    out = self.conv10(hs)
    return out
      
if __name__ == '__main__':
    model = WinNet(1, 2)
    total = sum([param.nelement() for param in model.parameters()])
    print("Numberof parameter: % .2fM" % (total / 1024**2))
    input = torch.randn(size=(2, 1,64, 64))
    out1= model(input)
    print(out1.shape)
import sys
from turtle import forward
sys.path.append('D:\\anaconda\\Lib\\site-packages')
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn, unsqueeze
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

# img=Image.open('1.jpeg')
# fig=plt.figure()
# plt.imshow(img)
# plt.show()

# transform=Compose([Resize((224,224)),ToTensor()])
# x=transform(img)
# x=x.unsqueeze(0)
# x.shape

# patch_size=16
# # 难点
# pathes=rearrange(x,'b c (h s1) (w s2) -> b (h w)(s1 s2 c)',s1=patch_size,s2=patch_size)
# pathes.shape

class PatchEmbedding(nn.Module):
    def __init__(self,in_channels:int=3,patch_size:int=16,emb_size:int=768,img_size:int=224):
        self.path_size=patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels,emb_size,kernel_size=patch_size,stride=patch_size),
            # 将卷积操作后的patch铺平，难点1
            Rearrange('b e (h) (w) -> b (h w) e')
            # break-down the image in s1 x s2 patches and flat them
            # Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            # 注意这里的隐层大小设置的也是768，可以配置
            # nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
        # 生成一个维度为emb_size的向量当做cls_token
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        # 位置编码信息,难点2
        self.positions=nn.Parameter(torch.randn((img_size // patch_size)**2 ,emb_size))

    def forward(self,x:Tensor) -> Tensor:
        # 获取batch大小
        b, _, _, _=x.shape       
        x=self.projection(x)
        # 将cls_token扩展b次
        # cls_tokens=repeat(self.cls_token,'() n e -> b n e', b=b)
        # x=torch.cat([cls_tokens,x],dim=1)
        # 将位置加到x上
        x+=self.positions
        # print(self.positions)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        # print("1qkv's shape: ", self.qkv(x).shape)
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        # print("2qkv's shape: ", qkv.shape)
        
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # print("queries's shape: ", queries.shape)
        # print("keys's shape: ", keys.shape)
        # print("values's shape: ", values.shape)
        
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        # print("energy's shape: ", energy.shape)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        # print("scaling: ", scaling)
        att = F.softmax(energy, dim=-1) / scaling
        # print("att1' shape: ", att.shape)
        att = self.att_drop(att)
        # print("att2' shape: ", att.shape)
        
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        # print("out1's shape: ", out.shape)
        out = rearrange(out, "b h n d -> b n (h d)")
        # print("out2's shape: ", out.shape)
        out = self.projection(out)
        # print("out3's shape: ", out.shape)
        return out
    


# PatchEmbedding()(x).shape

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                # nn.MultiheadAttention(emb_size,3),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 2):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes),
            nn.ReLU())
            
class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 3,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            Rearrange('b (h w) c -> b c h w',h=8,w=8)
            # Rearrange('b (c1 c2) n h w -> b n (c1 h) (c2 w)',c1=4,c2=4)
            # ClassificationHead(emb_size, n_classes)
        )


#自定义损失函数
class TripletLossFunc(nn.Module):
    def __init__(self,m:Tensor):
        
        super().__init__()
        self.m=m
    def forward(self,origin:Tensor,right:Tensor,wrong:Tensor):
        part1=F.pairwise_distance(origin,right)
        part2=torch.tensor(0)
        for i in range(wrong.shape[0]):
            part2=part2+max(0,(self.m-F.pairwise_distance(origin,wrong[i])))           
        return part1+part2


if __name__ == '__main__':
    model = ViT(in_channels=128,patch_size=4,emb_size=2048,img_size=16,depth=12,n_classes=2)
    total = sum([param.nelement() for param in model.parameters()])
    print("Numberof parameter: % .2fM" % (total / 1024**2))
    input = torch.randn(size=(2, 128,16, 16))
    out = model(input)
    print(out.shape)
    # print(out[:,1,:,:,:].shape)
# print(Reduce('b n e -> b e', reduction='mean')(TransformerEncoder()(PatchEmbedding()(x))).shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer_E(nn.Module):
    def __init__(self, dim, depth=2, heads=3, dim_head=16, mlp_dim=64, dropout = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TransformerEBlock(nn.Module):
    def __init__(self,in_c,out_c,depth):
        super().__init__()
        self.Embedding = nn.Sequential(
            nn.Linear(in_c,out_c)
        )
        self.transformer = nn.Sequential(
            Transformer_E(dim=out_c,depth=depth,mlp_dim=out_c//2,dropout=0.0),
        )
    def forward(self,X):
        H = X.size(2)
        E = rearrange(X, 'B c H W -> B (H W) c', H=H)
        E = self.Embedding(E)
        y1 = self.transformer(E)
        out = torch.add(E,y1)
        out = rearrange(out, 'B (H W) C -> B C H W', H=H)
        return out



class DownTrans(nn.Module):
    def __init__(self,in_c,out_c,depth):
        super().__init__()
        self.net = nn.Sequential(
            nn.AvgPool2d(2),
            TransformerEBlock(in_c, out_c,depth)
        )
    def forward(self,X):
        out = self.net(X)
        return out

class UpTrans(nn.Module):
    def __init__(self,in_c,mid_c,out_c,depth):
        super().__init__()
        self.trans = nn.Sequential(
            TransformerEBlock(in_c + mid_c, out_c,depth),
            nn.ConvTranspose2d(in_channels=out_c, out_channels=out_c, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU()
        )

    def forward(self,X,Y):
        x2 = torch.cat((X,Y),dim=1)
        out = self.trans(x2)
        return out


class GSNet(nn.Module):
    def __init__(self,hs_band,ms_band):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=hs_band+ms_band,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(),
        )

        self.conv = nn.Conv2d(in_channels=128+64,out_channels=hs_band,kernel_size=1,stride=1,padding=0)

        self.down1 = DownTrans(64,80,2)
        self.down2 = DownTrans(80,96,2)
        self.down3 = DownTrans(96,112,2)

        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels=112, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=2,stride=2,padding=0),
        )
        self.up2 = UpTrans(128,96,128,3)
        self.up3 = UpTrans(128,80,128,3)

    def forward(self, y,z):
        y_up = F.interpolate(y,scale_factor=8,mode='bicubic',align_corners=False)
        y1 = torch.cat((y_up,z),dim=1)
        y1 = self.conv1(y1)
        y2 = self.down1(y1)
        y3 = self.down2(y2)
        y4 = self.down3(y3)
        y4 = self.up1(y4)
        out1 = self.up2(y4,y3)
        out2 = self.up3(out1,y2)
        out3 = torch.cat((out2,y1),dim=1)
        out = self.conv(out3)
        return out


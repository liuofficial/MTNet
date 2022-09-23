import torch.nn as nn
import torch.nn.functional as F
import torch
import transformer

class Encoder_Decoder(nn.Module):
    def __init__(self,in_c):
        super(Encoder_Decoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_c, in_c,kernel_size=9,stride=8,padding=1,bias=False),
            nn.BatchNorm2d(in_c),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_c, in_c,kernel_size=9,stride=8,padding=1,output_padding=1,bias=False),
            nn.BatchNorm2d(in_c),
            nn.LeakyReLU(),
        )
    def forward(self, X,Y):
        out = self.encoder(X)
        out = out - Y
        out = self.decoder(out)
        return out

class TransBlock(nn.Module):
    def __init__(self, hs_band,pan_band,itra_num):
        super(TransBlock, self).__init__()
        self.net1 = Encoder_Decoder(hs_band)
        self.net2 = transformer.GSNet(hs_band,pan_band)
        self.itrator_num = itra_num
        self.t1 = nn.Parameter(torch.tensor(0.1))
        self.t2 = nn.Parameter(torch.tensor(0.1))
    def forward(self, Y,Z):
        X0 = F.interpolate(Y,scale_factor=8,mode='bicubic',align_corners=False)
        X = X0
        for i in range(self.itrator_num):
            # TransBlock
            Hi = self.net2(Y,Z)
            Gi = self.net1(X,Y)
            X = (1 - self.t2) * X - self.t1 * Gi + self.t2 * Hi
        return X
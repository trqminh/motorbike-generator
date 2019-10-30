import torch.nn as nn
import torch
from utils import *


class Generator(nn.Module):
    def __init__(self, nz=100, num_classes=1, channels=3, nfilt=64):
        super(Generator, self).__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.channels = channels
        
        self.label_emb = nn.Embedding(num_classes, nz)
        self.pixelnorm = PixelwiseNorm()
        
        # self.upconv0 = UpConvBlock(2*nz, nfilt*32, num_classes, k_size=4, stride=1, padding=0, norm="cbn", dropout_p=0.15)
        self.upconv1 = UpConvBlock(2*nz, nfilt*16, num_classes, k_size=4, stride=1, padding=0, norm="cbn", dropout_p=0.15)
        self.upconv2 = UpConvBlock(nfilt*16, nfilt*8, num_classes, k_size=4, stride=2, padding=1, norm="cbn", dropout_p=0.10)
        self.upconv3 = UpConvBlock(nfilt*8, nfilt*4, num_classes, k_size=4, stride=2, padding=1, norm="cbn", dropout_p=0.05)
        self.upconv4 = UpConvBlock(nfilt*4, nfilt*2, num_classes, k_size=4, stride=2, padding=1, norm="cbn", dropout_p=0.05)
        self.upconv5 = UpConvBlock(nfilt*2, nfilt, num_classes, k_size=4, stride=2, padding=1, norm="cbn", dropout_p=0.05)
        
        self.self_attn = Self_Attn(nfilt)
        self.upconv6 = UpConvBlock(nfilt, 3, num_classes, k_size=4, stride=2, padding=1, norm="cbn")
        self.out_conv = spectral_norm(nn.Conv2d(3, 3, 3, 1, 1, bias=False))  
        self.out_activ = nn.Tanh()
        
    def forward(self, inputs):
        z, labels = inputs
        
        enc = self.label_emb(labels).view((-1, self.nz, 1, 1))
        enc = F.normalize(enc, p=2, dim=1)
        x = torch.cat((z, enc), 1)
        
        x = self.upconv1((x, labels))
        x = self.upconv2((x, labels))
        x = self.upconv3((x, labels))
        x = self.upconv4((x, labels))
        x = self.upconv5((x, labels))
        x = self.self_attn(x)
        x = self.upconv6((x, labels))
        x = self.out_conv(x)
        img = self.out_activ(x)              
        return img
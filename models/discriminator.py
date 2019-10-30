import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from utils import *


class Discriminator(nn.Module):
    def __init__(self, num_classes=1, channels=3, nfilt=64):
        super(Discriminator, self).__init__()
        self.channels = channels
        self.num_classes = num_classes

        def down_convlayer(n_input, n_output, k_size=4, stride=2, padding=0, dropout_p=0.0):
            block = [spectral_norm(nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)),
                     nn.BatchNorm2d(n_output),
                     nn.LeakyReLU(0.2, inplace=True),
                    ]
            if dropout_p > 0.0: block.append(nn.Dropout(p=dropout_p))
            return block
        
        self.label_emb = nn.Embedding(num_classes, 128*128)
        self.model = nn.Sequential(
            *down_convlayer(self.channels + 1, nfilt, 4, 2, 1),
            Self_Attn(nfilt),
            
            *down_convlayer(nfilt, nfilt*2, 4, 2, 1, dropout_p=0.10),
            *down_convlayer(nfilt*2, nfilt*4, 4, 2, 1, dropout_p=0.15),
            *down_convlayer(nfilt*4, nfilt*8, 4, 2, 1, dropout_p=0.25),
            
            MinibatchStdDev(),
            spectral_norm(nn.Conv2d(nfilt*8 + 1, 1, 10, 1, 1, bias=False)),
        )

    def forward(self, inputs):
        imgs, labels = inputs
        enc = self.label_emb(labels).view((-1, 1, 128, 128))
        enc = F.normalize(enc, p=2, dim=1)
        x = torch.cat((imgs, enc), 1)   # 4 input feature maps(3rgb + 1label)
        
        out = self.model(x)
        out = out.view(-1)
        return out

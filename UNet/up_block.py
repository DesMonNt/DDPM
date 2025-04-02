import torch
import torch.nn as nn
from .residual_block import ResidualBlock

class UpBlock(nn.Module):
    def __init__(self, x_ch, skip_ch, out_ch, emb_ch):
        super().__init__()

        self.block = ResidualBlock(x_ch + skip_ch, out_ch, emb_ch)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, skip, emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)

        return self.block(x, emb)

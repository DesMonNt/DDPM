import torch.nn as nn
from .residual_block import ResidualBlock

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch):
        super().__init__()

        self.block = ResidualBlock(in_ch, out_ch, emb_ch)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, emb):
        x = self.block(x, emb)

        return self.pool(x), x
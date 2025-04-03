import torch
import torch.nn as nn
from .residual_block import ResidualBlock
from .attention_block import AttentionBlock

class UpBlock(nn.Module):
    def __init__(self, x_ch, skip_ch, out_ch, emb_ch, use_attention=False):
        super().__init__()

        self.use_attention = use_attention
        self.block = ResidualBlock(x_ch + skip_ch, out_ch, emb_ch)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        if self.use_attention:
            self.attention = AttentionBlock(out_ch)

    def forward(self, x, skip, emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, emb)

        if self.use_attention:
            x = self.attention(x)

        return x

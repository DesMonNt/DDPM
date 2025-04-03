import torch.nn as nn
from .residual_block import ResidualBlock
from .attention_block import AttentionBlock

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch, use_attention=False):
        super().__init__()

        self.use_attention = use_attention
        self.block = ResidualBlock(in_ch, out_ch, emb_ch)
        self.pool = nn.AvgPool2d(2)

        if use_attention:
            self.attention = AttentionBlock(out_ch)

    def forward(self, x, emb):
        x = self.block(x, emb)

        if self.use_attention:
            x = self.attention(x)

        return self.pool(x), x
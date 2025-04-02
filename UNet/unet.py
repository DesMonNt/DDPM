import torch.nn as nn
from .residual_block import ResidualBlock
from .up_block import UpBlock
from .down_block import DownBlock
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_ch=64, emb_ch=256, num_classes=None):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(32),
            nn.Linear(32, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch)
        )

        self.label_emb = nn.Embedding(num_classes, emb_ch) if num_classes is not None else None

        self.init_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)

        self.down1 = DownBlock(base_ch, base_ch * 2, emb_ch)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4, emb_ch)

        self.bot = ResidualBlock(base_ch * 4, base_ch * 4, emb_ch)

        self.up2 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2, emb_ch)
        self.up1 = UpBlock(base_ch * 2, base_ch * 2, base_ch, emb_ch)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, y=None):
        temb = self.time_mlp(t)

        if self.label_emb is not None and y is not None:
            yemb = self.label_emb(y)
            emb = temb + yemb
        else:
            emb = temb

        x = self.init_conv(x)
        x1 = x
        x2, skip1 = self.down1(x1, emb)
        x3, skip2 = self.down2(x2, emb)

        x_mid = self.bot(x3, emb)

        x = self.up2(x_mid, skip2, emb)
        x = self.up1(x, skip1, emb)

        return self.out_conv(x)

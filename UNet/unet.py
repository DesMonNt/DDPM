import torch.nn as nn
from .residual_block import ResidualBlock
from .up_block import UpBlock
from .down_block import DownBlock
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from .attention_block import AttentionBlock


class UNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            base_ch=64,
            emb_ch=256,
            num_classes=None,
            attention_levels=None,
            depth=2,
            image_size=32
    ):
        super().__init__()

        assert image_size % (2 ** depth) == 0

        self.depth = depth
        self.attention_levels = set(attention_levels) if attention_levels is not None else set()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(32),
            nn.Linear(32, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch)
        )
        self.label_emb = nn.Embedding(num_classes, emb_ch) if num_classes is not None else None

        self.init_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        in_ch = base_ch
        for i in range(depth):
            out_ch = base_ch * (2 ** (i + 1))
            self.downs.append(DownBlock(in_ch, out_ch, emb_ch, use_attention=i in self.attention_levels))
            in_ch = out_ch

        self.bot = ResidualBlock(in_ch, in_ch, emb_ch)
        self.bot_attention = AttentionBlock(in_ch) if self.depth in attention_levels else nn.Identity()

        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            skip_ch = base_ch * (2 ** (i + 1))
            out_ch = base_ch * (2 ** i)
            self.ups.append(UpBlock(in_ch, skip_ch, out_ch, emb_ch, use_attention=i in self.attention_levels))
            in_ch = out_ch

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, y=None):
        temb = self.time_mlp(t)
        emb = temb + self.label_emb(y) if self.label_emb is not None and y is not None else temb

        x = self.init_conv(x)

        skips = []
        for down in self.downs:
            x, skip = down(x, emb)
            skips.append(skip)

        x = self.bot(x, emb)
        x = self.bot_attention(x)

        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip, emb)

        return self.out_conv(x)

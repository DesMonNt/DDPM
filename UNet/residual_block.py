import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Linear(emb_ch, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(self.act1(self.norm1(x)))
        emb_out = self.emb_proj(emb)[:, :, None, None]
        h = h + emb_out
        h = self.conv2(self.act2(self.norm2(h)))

        return h + self.skip(x)
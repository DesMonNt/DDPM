import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attention(h, h, h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)

        return x + h
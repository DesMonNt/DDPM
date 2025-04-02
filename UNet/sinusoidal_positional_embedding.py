import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        exponent = -np.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * exponent)
        angles = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode='constant')

        return emb

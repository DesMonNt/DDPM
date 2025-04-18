import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import cosine_beta_schedule

class DDPM(nn.Module):
    def __init__(
        self,
        model,
        T=1000,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        num_classes=None
    ):
        super().__init__()

        self.model = model
        self.T = T
        self.num_classes = num_classes

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, T)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(T)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1. - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.alpha_bars[t][:, None, None, None].sqrt()
        sqrt_one_minus = (1 - self.alpha_bars[t])[:, None, None, None].sqrt()

        return sqrt_alpha_bar * x_0 + sqrt_one_minus * noise

    def training_step(self, x_0, y=None):
        B = x_0.size(0)
        t = torch.randint(0, self.T, (B,), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        pred_noise = self.model(x_t, t, y) if self.num_classes else self.model(x_t, t)
        loss = F.mse_loss(pred_noise, noise)

        return loss

    @torch.no_grad()
    def p_sample(self, x, t, y=None):
        betas_t = self.betas[t][:, None, None, None]
        alphas_t = self.alphas[t][:, None, None, None]
        alpha_bars_t = self.alpha_bars[t][:, None, None, None]

        pred_noise = self.model(x, t, y) if self.num_classes else self.model(x, t)

        coef1 = 1 / alphas_t.sqrt()
        coef2 = (1 - alphas_t) / (1 - alpha_bars_t).sqrt()
        mean = coef1 * (x - coef2 * pred_noise)

        noise = torch.randn_like(x)
        mask = (t > 0).float().view(-1, 1, 1, 1)
        sample = mean + mask * betas_t.sqrt() * noise

        return sample

    @torch.no_grad()
    def sample(self, label=None, shape=(1, 3, 32, 32)):
        x = torch.randn(shape, device=self.betas.device)

        if label is not None:
            y = torch.tensor([label] * shape[0], device=self.betas.device)
        else:
            y = None

        for t_ in reversed(range(self.T)):
            t = torch.full((shape[0],), t_, device=self.betas.device, dtype=torch.long)
            x = self.p_sample(x, t, y)

        return x
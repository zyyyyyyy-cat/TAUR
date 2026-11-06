"""TAUR-specific bounded patch residual; not a reproduced reference-paper module."""
import torch
from torch import nn


class ThermalAdapter(nn.Module):
    def __init__(self, dim=768, bottleneck=128, rho=0.10, eps=1e-6):
        super().__init__()
        if bottleneck <= 0 or not 0 <= rho <= 1:
            raise ValueError('Invalid TAUR bottleneck or residual bound')
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.rho, self.eps = rho, eps
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=0.001)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens, modality_ids):
        thermal = modality_ids == 3
        if not thermal.any():
            return tokens
        patches = tokens[thermal, 1:]
        residual = self.up(self.act(self.down(self.norm(patches))))
        # Norms are per image, in FP32 even inside AMP; do not detach either norm.
        h_norm = patches.float().norm(dim=(1, 2), keepdim=True)
        u_norm = residual.float().norm(dim=(1, 2), keepdim=True)
        scale = self.rho * h_norm / (h_norm + u_norm + self.eps)
        updated = patches + (scale * residual.float()).to(patches.dtype)
        result = tokens.clone()
        result[thermal, 1:] = updated
        return result

"""Broadcasting.

Smaller tensors stretch along missing or size-one axes so elementwise
operations line up: a bias per feature, a scale per token, and a table that
pairs every row with every column.
"""

import torch
from torch import nn

# track: Foundations
# input x: batch=2, tokens=4, features=6


class Broadcasting(nn.Module):
    def __init__(self, tokens=4, features=6):
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(-1, 1, features))  # one per feature
        self.scale = nn.Parameter(torch.linspace(0.5, 2, tokens).unsqueeze(-1))  # one per token

    def forward(self, x):
        shifted = x + self.bias  # [6] stretches over batch and tokens
        scaled = shifted * self.scale  # [4, 1] stretches over batch and features
        rows = torch.arange(4.0).unsqueeze(1)  # axes: tokens, one
        columns = torch.arange(6.0).unsqueeze(0)  # axes: one, features
        table = rows * 10 + columns  # axes: tokens, features
        kept = torch.where(table > 20, scaled, torch.zeros_like(scaled))
        return kept

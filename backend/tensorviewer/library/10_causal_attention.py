"""Causal attention.

A language model may not peek ahead. A lower-triangular mask sets every score
for a later token to minus infinity, so softmax gives it exactly zero weight.
"""

import math

import torch
from torch import nn

# track: Attention
# input x: batch=1, tokens=6, features=8


class CausalAttention(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        tokens = x.shape[1]
        q = self.query(x)  # axes: batch, tokens, features
        k = self.key(x)  # axes: batch, tokens, features
        v = self.value(x)  # axes: batch, tokens, features
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # axes: batch, queries, keys
        allowed = torch.ones(tokens, tokens).tril().bool()  # axes: queries, keys
        scores = scores.masked_fill(~allowed, float("-inf"))  # axes: batch, queries, keys
        weights = scores.softmax(dim=-1)  # axes: batch, queries, keys
        return weights @ v  # axes: batch, tokens, features

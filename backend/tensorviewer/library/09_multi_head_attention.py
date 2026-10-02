"""Multi-head attention.

The features split into heads that attend independently, each with its own
queries and keys. The heads' outputs are put back side by side and mixed by
an output projection.
"""

import math

from torch import nn

# track: Attention
# input x: batch=2, tokens=5, features=8


class MultiHeadAttention(nn.Module):
    def __init__(self, dim=8, heads=2):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)

    def split(self, x):
        batch, tokens, dim = x.shape
        x = x.reshape(batch, tokens, self.heads, dim // self.heads)  # axes: batch, tokens, heads, head_features
        return x.transpose(1, 2)  # axes: batch, heads, tokens, head_features

    def forward(self, x):
        batch, tokens, dim = x.shape
        q = self.split(self.query(x))
        k = self.split(self.key(x))
        v = self.split(self.value(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # axes: batch, heads, queries, keys
        weights = scores.softmax(dim=-1)  # axes: batch, heads, queries, keys
        heads = weights @ v  # axes: batch, heads, tokens, head_features
        merged = heads.transpose(1, 2).reshape(batch, tokens, dim)  # axes: batch, tokens, features
        return self.output(merged)

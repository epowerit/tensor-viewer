"""Scaled dot-product attention.

Every token scores every other token with a dot product of query and key.
Softmax turns each row of scores into weights, and the weights mix the value
vectors into the output.
"""

import math

from torch import nn

# track: Attention
# input x: batch=1, tokens=6, features=8


class Attention(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q = self.query(x)  # axes: batch, tokens, features
        k = self.key(x)  # axes: batch, tokens, features
        v = self.value(x)  # axes: batch, tokens, features
        scores = q @ k.transpose(-2, -1)  # axes: batch, queries, keys
        scores = scores / math.sqrt(q.shape[-1])  # axes: batch, queries, keys
        weights = scores.softmax(dim=-1)  # axes: batch, queries, keys
        return weights @ v  # axes: batch, tokens, features

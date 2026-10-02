"""Perceiver.

Attention over every pixel grows with the square of the image. The Perceiver
instead keeps a small array of learned latents: they cross-attend to the
whole input once, then only the latents attend to each other, block after
block. The cost no longer depends on how large the input is.
"""

import math

import torch
from torch import nn

# track: Vision transformers
# input image: batch=1, channels=3, height=16, width=16 | image


class Attention(nn.Module):
    """Queries from one sequence, keys and values from another (or the same)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(dim, dim)
        self.key_value = nn.Linear(dim, 2 * dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x, context):
        batch, tokens, dim = x.shape
        length = context.shape[1]
        q = self.query(x).reshape(batch, tokens, self.heads, -1).transpose(1, 2)  # axes: batch, heads, queries, head_features
        k, v = self.key_value(context).chunk(2, dim=-1)
        k = k.reshape(batch, length, self.heads, -1).transpose(1, 2)  # axes: batch, heads, keys, head_features
        v = v.reshape(batch, length, self.heads, -1).transpose(1, 2)  # axes: batch, heads, keys, head_features
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)  # axes: batch, heads, queries, keys
        mixed = (weights @ v).transpose(1, 2).reshape(batch, tokens, dim)  # axes: batch, queries, features
        return self.output(mixed)


class LatentBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = Attention(dim, heads)
        self.feed_forward = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, latents):
        normed = self.norm(latents)  # axes: batch, latents, features
        latents = latents + self.attention(normed, normed)  # axes: batch, latents, features
        return latents + self.feed_forward(latents)


class Perceiver(nn.Module):
    def __init__(self, channels=3, dim=16, latents=8, heads=2, depth=2, classes=10, pixels=256):
        super().__init__()
        self.embed = nn.Linear(channels, dim)
        self.position = nn.Parameter(torch.randn(1, pixels, dim) * 0.02)
        self.latents = nn.Parameter(torch.randn(1, latents, dim) * 0.02)
        self.read_norm = nn.LayerNorm(dim)
        self.read = Attention(dim, heads)
        self.blocks = nn.ModuleList(LatentBlock(dim, heads) for _ in range(depth))
        self.head = nn.Linear(dim, classes)

    def forward(self, image):
        inputs = image.flatten(2).transpose(1, 2)  # axes: batch, pixels, channels
        inputs = self.embed(inputs) + self.position  # axes: batch, pixels, features
        latents = self.latents.expand(image.shape[0], -1, -1)  # axes: batch, latents, features
        # One read of the whole input: 8 latents attend to 256 pixels.
        latents = latents + self.read(self.read_norm(latents), inputs)  # axes: batch, latents, features
        for block in self.blocks:
            latents = block(latents)
        return self.head(latents.mean(dim=1)).softmax(dim=-1)  # axes: batch, classes

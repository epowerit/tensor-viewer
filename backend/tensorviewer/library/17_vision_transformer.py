"""Vision Transformer.

ViT treats an image as a sentence of patches. Sixteen patch tokens and a class
token get position embeddings, two transformer blocks let every patch attend
to every other, and the class token's final vector is classified.
"""

import math

import torch
from torch import nn

# track: Vision transformers
# input image: batch=2, channels=3, height=16, width=16 | image


class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x):
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(batch, tokens, self.heads, -1).transpose(1, 2)  # axes: batch, heads, tokens, head_features
        k = k.reshape(batch, tokens, self.heads, -1).transpose(1, 2)  # axes: batch, heads, tokens, head_features
        v = v.reshape(batch, tokens, self.heads, -1).transpose(1, 2)  # axes: batch, heads, tokens, head_features
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # axes: batch, heads, queries, keys
        weights = scores.softmax(dim=-1)  # axes: batch, heads, queries, keys
        mixed = (weights @ v).transpose(1, 2).reshape(batch, tokens, dim)  # axes: batch, tokens, features
        return self.output(mixed)


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size=16, patch=4, channels=3, dim=16, heads=2, depth=2, classes=10):
        super().__init__()
        patches = (image_size // patch) ** 2
        self.patch_embedding = nn.Conv2d(channels, dim, kernel_size=patch, stride=patch)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.randn(1, patches + 1, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, image):
        grid = self.patch_embedding(image)  # axes: batch, features, rows, columns
        x = grid.flatten(2).transpose(1, 2)  # axes: batch, patches, features
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1)  # axes: batch, tokens, features
        x = x + self.position
        for block in self.blocks:
            x = block(x)
        summary = self.norm(x[:, 0])  # axes: batch, features
        return self.head(summary).softmax(dim=-1)

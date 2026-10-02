"""Swin Transformer.

A hierarchical vision transformer. Attention runs inside small windows to stay
cheap; the second block shifts the windows with a roll so information crosses
their borders, masking pairs that only touch because of the wrap-around. Patch
merging then halves the resolution and doubles the features.
"""

import math

import torch
from torch import nn

# track: Vision transformers
# input image: batch=1, channels=3, height=16, width=16 | image


def partition(x, window):
    """[batch, height, width, features] -> [windows, window * window, features]"""
    batch, height, width, features = x.shape
    x = x.reshape(batch, height // window, window, width // window, window, features)
    x = x.permute(0, 1, 3, 2, 4, 5)  # axes: batch, window_rows, window_columns, rows, columns, features
    return x.reshape(-1, window * window, features)


def reverse(windows, window, batch, height, width):
    """[windows, window * window, features] -> [batch, height, width, features]"""
    features = windows.shape[-1]
    x = windows.reshape(batch, height // window, width // window, window, window, features)
    x = x.permute(0, 1, 3, 2, 4, 5)  # axes: batch, window_rows, rows, window_columns, columns, features
    return x.reshape(batch, height, width, features)


class WindowAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        windows, tokens, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(windows, tokens, self.heads, -1).transpose(1, 2)  # axes: windows, heads, tokens, head_features
        k = k.reshape(windows, tokens, self.heads, -1).transpose(1, 2)  # axes: windows, heads, tokens, head_features
        v = v.reshape(windows, tokens, self.heads, -1).transpose(1, 2)  # axes: windows, heads, tokens, head_features
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # axes: windows, heads, queries, keys
        if mask is not None:
            scores = scores + mask.unsqueeze(1)
        weights = scores.softmax(dim=-1)  # axes: windows, heads, queries, keys
        mixed = (weights @ v).transpose(1, 2).reshape(windows, tokens, dim)  # axes: windows, tokens, features
        return self.output(mixed)


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, resolution, window, shift):
        super().__init__()
        self.window, self.shift = window, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        if shift:
            # Label the regions that the roll glues together; tokens from
            # different regions must not attend to each other.
            regions = torch.zeros(1, resolution, resolution, 1)
            label = 0
            for rows in (slice(0, -window), slice(-window, -shift), slice(-shift, None)):
                for columns in (slice(0, -window), slice(-window, -shift), slice(-shift, None)):
                    regions[:, rows, columns, :] = label
                    label += 1
            labels = partition(regions, window).squeeze(-1)
            mask = labels.unsqueeze(1) - labels.unsqueeze(2)
            self.register_buffer("mask", mask.masked_fill(mask != 0, -100.0))
        else:
            self.mask = None

    def forward(self, x):
        batch, height, width, _ = x.shape
        shortcut = x
        x = self.norm1(x)
        if self.shift:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        windows = partition(x, self.window)  # axes: windows, tokens, features
        mask = None if self.mask is None else self.mask.repeat(batch, 1, 1)
        attended = self.attention(windows, mask)  # axes: windows, tokens, features
        x = reverse(attended, self.window, batch, height, width)  # axes: batch, height, width, features
        if self.shift:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        corners = [x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]]
        merged = torch.cat(corners, dim=-1)  # axes: batch, height, width, features
        return self.reduction(self.norm(merged))


class SwinTransformer(nn.Module):
    def __init__(self, image_size=16, patch=2, channels=3, dim=12, heads=2, window=4, classes=10):
        super().__init__()
        resolution = image_size // patch
        self.patch_embedding = nn.Conv2d(channels, dim, kernel_size=patch, stride=patch)
        self.regular = SwinBlock(dim, heads, resolution, window, shift=0)
        self.shifted = SwinBlock(dim, heads, resolution, window, shift=window // 2)
        self.merge = PatchMerging(dim)
        self.norm = nn.LayerNorm(2 * dim)
        self.head = nn.Linear(2 * dim, classes)

    def forward(self, image):
        x = self.patch_embedding(image)  # axes: batch, features, height, width
        x = x.permute(0, 2, 3, 1)  # axes: batch, height, width, features
        x = self.regular(x)
        x = self.shifted(x)
        x = self.merge(x)  # axes: batch, height, width, features
        pooled = self.norm(x.mean(dim=(1, 2)))  # axes: batch, features
        return self.head(pooled).softmax(dim=-1)

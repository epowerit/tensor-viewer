"""Object detection transformer.

DETR treats detection as predicting a set. A small CNN turns the image into a
feature map whose pixels become tokens for a transformer encoder. A fixed set
of learned object queries then cross-attends to those tokens in a decoder,
and every query predicts one class (or "no object") and one box.
"""

import math

import torch
from torch import nn

# track: Vision transformers
# input image: batch=1, channels=3, height=32, width=32 | image


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


class DETR(nn.Module):
    def __init__(self, dim=16, heads=2, queries=6, classes=4, grid=8):
        super().__init__()
        # Two strided convolutions shrink 32×32 pixels to an 8×8 feature map.
        self.backbone = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
        )
        self.position = nn.Parameter(torch.randn(1, grid * grid, dim) * 0.02)
        self.encoder_norm = nn.LayerNorm(dim)
        self.encoder = Attention(dim, heads)
        self.queries = nn.Parameter(torch.randn(1, queries, dim) * 0.02)
        self.decoder_norm = nn.LayerNorm(dim)
        self.cross_attention = Attention(dim, heads)
        self.feed_forward = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))
        # One extra class means "no object".
        self.classify = nn.Linear(dim, classes + 1)
        self.box = nn.Linear(dim, 4)

    def forward(self, image):
        features = self.backbone(image)  # axes: batch, features, rows, columns
        memory = features.flatten(2).transpose(1, 2) + self.position  # axes: batch, pixels, features
        normed = self.encoder_norm(memory)  # axes: batch, pixels, features
        memory = memory + self.encoder(normed, normed)  # axes: batch, pixels, features
        objects = self.queries.expand(image.shape[0], -1, -1)  # axes: batch, queries, features
        objects = objects + self.cross_attention(self.decoder_norm(objects), memory)  # axes: batch, queries, features
        objects = objects + self.feed_forward(objects)  # axes: batch, queries, features
        classes = self.classify(objects).softmax(dim=-1)  # axes: batch, queries, classes
        boxes = self.box(objects).sigmoid()  # axes: batch, queries, box
        return classes, boxes

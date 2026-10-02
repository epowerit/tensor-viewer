"""Image captioner.

A vision encoder and a language decoder joined by cross-attention. A ViT
turns the image into patch tokens; the decoder reads the caption so far with
causal self-attention, then looks at the patches through cross-attention, and
scores the next word at every position.
"""

import math

import torch
from torch import nn

# track: Multimodal
# input image: batch=1, channels=3, height=16, width=16 | image
# input caption: text "a bright disc on a soft gradient"


class Attention(nn.Module):
    """Queries from one sequence, keys and values from another (or the same)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(dim, dim)
        self.key_value = nn.Linear(dim, 2 * dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x, context, causal=False):
        batch, tokens, dim = x.shape
        length = context.shape[1]
        q = self.query(x).reshape(batch, tokens, self.heads, -1).transpose(1, 2)  # axes: batch, heads, queries, head_features
        k, v = self.key_value(context).chunk(2, dim=-1)
        k = k.reshape(batch, length, self.heads, -1).transpose(1, 2)  # axes: batch, heads, keys, head_features
        v = v.reshape(batch, length, self.heads, -1).transpose(1, 2)  # axes: batch, heads, keys, head_features
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # axes: batch, heads, queries, keys
        if causal:
            allowed = torch.ones(tokens, length).tril().bool()  # axes: queries, keys
            scores = scores.masked_fill(~allowed, float("-inf"))
        weights = scores.softmax(dim=-1)  # axes: batch, heads, queries, keys
        mixed = (weights @ v).transpose(1, 2).reshape(batch, tokens, dim)  # axes: batch, queries, features
        return self.output(mixed)


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attention = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attention = Attention(dim, heads)
        self.norm3 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, words, patches):
        normed = self.norm1(words)  # axes: batch, words, features
        words = words + self.self_attention(normed, normed, causal=True)  # axes: batch, words, features
        words = words + self.cross_attention(self.norm2(words), patches)  # axes: batch, words, features
        return words + self.feed_forward(self.norm3(words))


class ImageCaptioner(nn.Module):
    def __init__(self, image_size=16, patch=4, channels=3, dim=16, heads=2, depth=2, vocabulary=64, max_length=64):
        super().__init__()
        self.patch_embedding = nn.Conv2d(channels, dim, kernel_size=patch, stride=patch)
        self.patch_position = nn.Parameter(torch.randn(1, (image_size // patch) ** 2, dim) * 0.02)
        self.encoder_norm = nn.LayerNorm(dim)
        self.encoder = Attention(dim, heads)
        self.token = nn.Embedding(vocabulary, dim)
        self.position = nn.Embedding(max_length, dim)
        self.blocks = nn.ModuleList(DecoderBlock(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocabulary)

    def forward(self, image, caption):
        patches = self.patch_embedding(image).flatten(2).transpose(1, 2) + self.patch_position  # axes: batch, patches, features
        normed = self.encoder_norm(patches)  # axes: batch, patches, features
        patches = patches + self.encoder(normed, normed)  # axes: batch, patches, features
        positions = torch.arange(caption.shape[1]).unsqueeze(0)  # axes: batch, words
        words = self.token(caption) + self.position(positions)  # axes: batch, words, features
        for block in self.blocks:
            words = block(words, patches)
        return self.head(self.norm(words)).softmax(dim=-1)  # axes: batch, words, vocabulary

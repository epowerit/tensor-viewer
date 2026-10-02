"""Transformer encoder classifier.

A BERT-style encoder. A learned class token joins the sentence, positions are
added, and two pre-norm blocks of self-attention and feed-forward layers mix
every token with every other. The class token's final vector is classified.
"""

import math

import torch
from torch import nn

# track: Text transformers
# input tokens: text "this movie was a delight from the first scene to the last"


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


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class EncoderClassifier(nn.Module):
    def __init__(self, vocabulary=64, dim=16, heads=2, depth=2, classes=2, max_length=65):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.randn(1, max_length, dim) * 0.02)
        self.blocks = nn.ModuleList(EncoderBlock(dim, heads, 4 * dim) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, tokens):
        x = self.embedding(tokens)  # axes: batch, tokens, features
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1)  # axes: batch, tokens, features
        x = x + self.position[:, : x.shape[1]]
        for block in self.blocks:
            x = block(x)
        summary = self.norm(x[:, 0])  # axes: batch, features
        return self.head(summary).softmax(dim=-1)

"""Encoder-decoder translator.

The original transformer. The encoder reads the source sentence; the decoder
writes the target one position at a time, with causal self-attention over
what it has written and cross-attention into the encoded source.
"""

import math

import torch
from torch import nn

# track: Text transformers
# input source: text "the cat sat on the mat"
# input target: text "le chat est assis sur le tapis"


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


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attention = Attention(dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x + self.attention(x, x))
        return self.norm2(x + self.feed_forward(x))


class Decoder(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.self_attention = Attention(dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attention = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, y, memory):
        y = self.norm1(y + self.self_attention(y, y, causal=True))
        y = self.norm2(y + self.cross_attention(y, memory))
        return self.norm3(y + self.feed_forward(y))


class Translator(nn.Module):
    def __init__(self, vocabulary=64, dim=16, heads=2, max_length=64):
        super().__init__()
        self.source_embedding = nn.Embedding(vocabulary, dim)
        self.target_embedding = nn.Embedding(vocabulary, dim)
        self.position = nn.Embedding(max_length, dim)
        self.encoder = Encoder(dim, heads)
        self.decoder = Decoder(dim, heads)
        self.head = nn.Linear(dim, vocabulary)

    def embed(self, tokens, table):
        positions = torch.arange(tokens.shape[1]).unsqueeze(0)  # axes: batch, tokens
        return table(tokens) + self.position(positions)  # axes: batch, tokens, features

    def forward(self, source, target):
        memory = self.encoder(self.embed(source, self.source_embedding))  # axes: batch, source_tokens, features
        decoded = self.decoder(self.embed(target, self.target_embedding), memory)  # axes: batch, target_tokens, features
        logits = self.head(decoded)  # axes: batch, target_tokens, vocabulary
        return logits.softmax(dim=-1)

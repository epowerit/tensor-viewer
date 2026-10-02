"""GPT language model.

A decoder-only transformer. Causal self-attention lets every token look only
backward; after two blocks, the output layer, tied to the token embedding,
scores the next token at every position.
"""

import math

import torch
from torch import nn

# track: Text transformers
# input tokens: text "once upon a time there was a small robot who loved to read"


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads, max_length):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.register_buffer("causal", torch.ones(max_length, max_length).tril().bool())

    def forward(self, x):
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.reshape(batch, tokens, self.heads, -1).transpose(1, 2)  # axes: batch, heads, tokens, head_features
        k = k.reshape(batch, tokens, self.heads, -1).transpose(1, 2)  # axes: batch, heads, tokens, head_features
        v = v.reshape(batch, tokens, self.heads, -1).transpose(1, 2)  # axes: batch, heads, tokens, head_features
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # axes: batch, heads, queries, keys
        allowed = self.causal[:tokens, :tokens]  # axes: queries, keys
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = scores.softmax(dim=-1)  # axes: batch, heads, queries, keys
        mixed = (weights @ v).transpose(1, 2).reshape(batch, tokens, dim)  # axes: batch, tokens, features
        return self.output(mixed)


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, max_length):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = CausalSelfAttention(dim, heads, max_length)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocabulary=64, dim=16, heads=2, depth=2, max_length=64):
        super().__init__()
        self.token = nn.Embedding(vocabulary, dim)
        self.position = nn.Embedding(max_length, dim)
        self.blocks = nn.ModuleList(DecoderBlock(dim, heads, max_length) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1]).unsqueeze(0)  # axes: batch, tokens
        x = self.token(tokens) + self.position(positions)  # axes: batch, tokens, features
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = x @ self.token.weight.T  # axes: batch, tokens, vocabulary
        return logits.softmax(dim=-1)

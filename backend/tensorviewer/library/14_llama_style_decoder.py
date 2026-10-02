"""LLaMA-style decoder.

The block design of modern open language models. RMSNorm replaces LayerNorm,
rotary position embeddings (RoPE) rotate queries and keys by their position
instead of adding a position vector, and a gated SwiGLU feed-forward replaces
the GELU MLP.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

# track: Text transformers
# input tokens: text "rotary embeddings encode position by turning each query and key"


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # axes: batch, tokens, one
        return x * scale * self.weight


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


class RotaryAttention(nn.Module):
    def __init__(self, dim, heads, max_length):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        frequency = 1.0 / 10000 ** (torch.arange(0, head_dim, 2) / head_dim)
        angles = torch.arange(max_length).unsqueeze(1) * frequency
        angles = torch.cat([angles, angles], dim=-1)
        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())
        self.register_buffer("causal", torch.ones(max_length, max_length).tril().bool())

    def forward(self, x):
        batch, tokens, dim = x.shape
        q = self.query(x).reshape(batch, tokens, self.heads, -1).transpose(1, 2)  # axes: batch, heads, tokens, head_features
        k = self.key(x).reshape(batch, tokens, self.heads, -1).transpose(1, 2)  # axes: batch, heads, tokens, head_features
        v = self.value(x).reshape(batch, tokens, self.heads, -1).transpose(1, 2)  # axes: batch, heads, tokens, head_features
        cos, sin = self.cos[:tokens], self.sin[:tokens]
        q = q * cos + rotate_half(q) * sin  # axes: batch, heads, tokens, head_features
        k = k * cos + rotate_half(k) * sin  # axes: batch, heads, tokens, head_features
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # axes: batch, heads, queries, keys
        scores = scores.masked_fill(~self.causal[:tokens, :tokens], float("-inf"))
        weights = scores.softmax(dim=-1)  # axes: batch, heads, queries, keys
        mixed = (weights @ v).transpose(1, 2).reshape(batch, tokens, dim)  # axes: batch, tokens, features
        return self.output(mixed)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        gated = F.silu(self.gate(x)) * self.up(x)  # axes: batch, tokens, hidden
        return self.down(gated)


class LlamaBlock(nn.Module):
    def __init__(self, dim, heads, max_length):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.attention = RotaryAttention(dim, heads, max_length)
        self.feed_forward_norm = RMSNorm(dim)
        self.feed_forward = SwiGLU(dim, 2 * dim)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class TinyLlama(nn.Module):
    def __init__(self, vocabulary=64, dim=16, heads=2, depth=2, max_length=64):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, dim)
        self.blocks = nn.ModuleList(LlamaBlock(dim, heads, max_length) for _ in range(depth))
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocabulary, bias=False)

    def forward(self, tokens):
        x = self.embedding(tokens)  # axes: batch, tokens, features
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))  # axes: batch, tokens, vocabulary
        return logits.softmax(dim=-1)

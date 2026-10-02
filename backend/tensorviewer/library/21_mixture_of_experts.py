"""Mixture of experts.

A transformer block whose feed-forward layer is a team of experts. A router
scores every token against each expert and picks its best one (top-1 routing,
as in the Switch Transformer); the token's output is that expert's output,
scaled by the router's confidence. For a clear picture this model runs every
expert and gathers the chosen one; large models dispatch each token so only
its expert runs.
"""

import math

import torch
from torch import nn

# track: Text transformers
# input tokens: text "the router sends every token to the expert it trusts most"


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
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)  # axes: batch, heads, queries, keys
        mixed = (weights @ v).transpose(1, 2).reshape(batch, tokens, dim)  # axes: batch, tokens, features
        return self.output(mixed)


class Experts(nn.Module):
    """Top-1 routing over a few feed-forward experts."""

    def __init__(self, dim, experts, hidden):
        super().__init__()
        self.router = nn.Linear(dim, experts)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
            for _ in range(experts)
        )

    def forward(self, x):
        batch, tokens, dim = x.shape
        flat = x.reshape(batch * tokens, dim)  # axes: tokens, features
        probabilities = self.router(flat).softmax(dim=-1)  # axes: tokens, experts
        confidence, choice = probabilities.max(dim=-1)  # axes: tokens
        outputs = []
        for expert in self.experts:
            outputs.append(expert(flat))  # axes: tokens, features
        stacked = torch.stack(outputs, dim=1)  # axes: tokens, experts, features
        index = choice.reshape(-1, 1, 1).expand(-1, 1, dim)  # axes: tokens, one, features
        chosen = stacked.gather(1, index).squeeze(1)  # axes: tokens, features
        routed = chosen * confidence.unsqueeze(-1)  # axes: tokens, features
        return routed.reshape(batch, tokens, dim)  # axes: batch, tokens, features


class MixtureOfExperts(nn.Module):
    def __init__(self, vocabulary=64, dim=16, heads=2, experts=4, max_length=64):
        super().__init__()
        self.token = nn.Embedding(vocabulary, dim)
        self.position = nn.Embedding(max_length, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = SelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.experts = Experts(dim, experts, 2 * dim)
        self.head = nn.Linear(dim, vocabulary)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1]).unsqueeze(0)  # axes: batch, tokens
        x = self.token(tokens) + self.position(positions)  # axes: batch, tokens, features
        x = x + self.attention(self.norm1(x))
        x = x + self.experts(self.norm2(x))
        return self.head(x).softmax(dim=-1)  # axes: batch, tokens, vocabulary

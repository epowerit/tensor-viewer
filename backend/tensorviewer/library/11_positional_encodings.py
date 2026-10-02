"""Positional encodings.

Attention alone ignores word order, so every position adds its own vector to
the token embedding: fixed sinusoids, as in the original transformer, or a
learned table, as in BERT and GPT.
"""

import math

import torch
from torch import nn

# track: Attention
# input tokens: text "attention needs to know where each word sits"


class PositionalEncodings(nn.Module):
    def __init__(self, vocabulary=64, dim=16, max_length=64):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, dim)
        position = torch.arange(max_length).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        table = torch.zeros(max_length, dim)
        table[:, 0::2] = torch.sin(position * frequency)
        table[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("sinusoids", table)
        self.learned = nn.Embedding(max_length, dim)

    def forward(self, tokens):
        length = tokens.shape[1]
        words = self.embedding(tokens)  # axes: batch, tokens, features
        fixed = words + self.sinusoids[:length]  # axes: batch, tokens, features
        positions = torch.arange(length).unsqueeze(0)  # axes: batch, tokens
        learned = words + self.learned(positions)  # axes: batch, tokens, features
        return fixed, learned

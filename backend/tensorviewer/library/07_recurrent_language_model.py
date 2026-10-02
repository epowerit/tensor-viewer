"""Recurrent language model.

A GRU reads the sentence one token at a time, carrying a hidden state forward.
At every position, a linear layer turns that state into a guess for the next
token.
"""

from torch import nn

# track: Foundations
# input tokens: text "the quick brown fox jumps over the lazy dog"


class RecurrentLM(nn.Module):
    def __init__(self, vocabulary=64, dim=16, hidden=24):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, dim)
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocabulary)

    def forward(self, tokens):
        x = self.embedding(tokens)  # axes: batch, tokens, features
        states, _ = self.gru(x)  # axes: batch, tokens, hidden
        logits = self.head(states)  # axes: batch, tokens, vocabulary
        return logits.softmax(dim=-1)

"""Multilayer perceptron.

Two linear layers with a GELU between them turn 16 input features into
probabilities over 10 classes. Every output feature is a weighted sum of all
input features.
"""

from torch import nn

# track: Foundations
# input x: batch=4, features=16


class MLP(nn.Module):
    def __init__(self, features=16, hidden=32, classes=10):
        super().__init__()
        self.hidden = nn.Linear(features, hidden)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden, classes)

    def forward(self, x):
        hidden = self.hidden(x)  # axes: batch, hidden
        hidden = self.activation(hidden)  # axes: batch, hidden
        logits = self.output(hidden)  # axes: batch, classes
        return logits.softmax(dim=-1)

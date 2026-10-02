"""Reductions and softmax.

Sum, mean, and max collapse an axis into one number per group. Softmax keeps
the axis and turns scores into weights that sum to one.
"""

from torch import nn

# track: Foundations
# input scores: batch=2, tokens=4, classes=5


class Reductions(nn.Module):
    def forward(self, scores):
        total = scores.sum(dim=-1)  # axes: batch, tokens
        average = scores.mean(dim=1)  # axes: batch, classes
        best = scores.amax(dim=-1)  # axes: batch, tokens
        weights = scores.softmax(dim=-1)  # axes: batch, tokens, classes
        check = weights.sum(dim=-1)  # every row of weights sums to one
        log_weights = scores.log_softmax(dim=-1)  # axes: batch, tokens, classes
        return total, average, best, weights, check, log_weights

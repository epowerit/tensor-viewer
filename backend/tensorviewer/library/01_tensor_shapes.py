"""Tensor shapes.

Reshape, permute, unsqueeze, and flatten move the same 24 numbers into new
arrangements; nothing is computed. Follow one value through every view.
"""

from torch import nn

# track: Foundations
# input x: batch=2, rows=3, columns=4 | arange


class TensorShapes(nn.Module):
    def forward(self, x):
        cells = x.reshape(2, 12)  # axes: batch, cells
        columns_first = x.permute(0, 2, 1)  # axes: batch, columns, rows
        layered = x.unsqueeze(1)  # axes: batch, layer, rows, columns
        flat = columns_first.flatten(1)  # axes: batch, cells
        halves = x.reshape(2, 3, 2, 2)  # axes: batch, rows, half, columns
        return cells, columns_first, layered, flat, halves

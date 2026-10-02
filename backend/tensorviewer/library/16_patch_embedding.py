"""Patch embedding.

A vision transformer first cuts the image into square patches and projects
each patch to a token. A strided convolution does this in one step; cutting
the patches out with unfold and multiplying by the same weights gives the
same tokens.
"""

import torch.nn.functional as F
from torch import nn

# track: Vision transformers
# input image: batch=1, channels=3, height=16, width=16 | image


class PatchEmbedding(nn.Module):
    def __init__(self, channels=3, patch=4, dim=12):
        super().__init__()
        self.patch = patch
        self.projection = nn.Conv2d(channels, dim, kernel_size=patch, stride=patch)

    def forward(self, image):
        grid = self.projection(image)  # axes: batch, features, rows, columns
        tokens = grid.flatten(2).transpose(1, 2)  # axes: batch, patches, features
        # The same tokens without a convolution: cut out patches, then project.
        patches = F.unfold(image, self.patch, stride=self.patch)  # axes: batch, pixels, patches
        weight = self.projection.weight.flatten(1)  # axes: features, pixels
        direct = patches.transpose(1, 2) @ weight.T + self.projection.bias  # axes: batch, patches, features
        return tokens, direct

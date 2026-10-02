"""Convolutional image classifier.

Small kernels slide over the image to detect local patterns; pooling halves
the resolution; a linear layer reads the final feature map and scores 10
classes.
"""

import torch
from torch import nn

# track: Foundations
# input image: batch=2, channels=3, height=16, width=16 | image


class ConvNet(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Linear(16 * 4 * 4, classes)

    def forward(self, image):
        x = torch.relu(self.conv1(image))  # axes: batch, channels, height, width
        x = self.pool(x)  # axes: batch, channels, height, width
        x = torch.relu(self.conv2(x))  # axes: batch, channels, height, width
        x = self.pool(x)  # axes: batch, channels, height, width
        features = x.flatten(1)  # axes: batch, features
        return self.head(features).softmax(dim=-1)

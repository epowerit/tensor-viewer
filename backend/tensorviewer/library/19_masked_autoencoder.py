"""Masked autoencoder.

MAE pretraining hides most image patches. The encoder sees only the visible
quarter; a light decoder fills the hidden places with a learned mask token,
reconstructs every patch's pixels, and the error counts only where pixels were
hidden.
"""

import math

import torch
from torch import nn

# track: Vision transformers
# input image: batch=1, channels=3, height=16, width=16 | image


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x):
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.reshape(batch, tokens, self.heads, -1).transpose(1, 2)  # axes: batch, heads, tokens, head_features
        k = k.reshape(batch, tokens, self.heads, -1).transpose(1, 2)  # axes: batch, heads, tokens, head_features
        v = v.reshape(batch, tokens, self.heads, -1).transpose(1, 2)  # axes: batch, heads, tokens, head_features
        weights = (q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(dim=-1)
        mixed = (weights @ v).transpose(1, 2).reshape(batch, tokens, dim)  # axes: batch, tokens, features
        x = x + self.output(mixed)
        return x + self.mlp(self.norm2(x))


class MaskedAutoencoder(nn.Module):
    def __init__(self, image_size=16, patch=4, channels=3, dim=16, decoder_dim=12, keep_every=4):
        super().__init__()
        self.patch = patch
        self.patches = (image_size // patch) ** 2
        pixels = patch * patch * channels
        indices = range(self.patches)
        self.register_buffer("visible", torch.tensor([i for i in indices if i % keep_every == 0]))
        self.register_buffer("hidden", torch.tensor([i for i in indices if i % keep_every != 0]))
        self.embed = nn.Linear(pixels, dim)
        self.position = nn.Parameter(torch.randn(1, self.patches, dim) * 0.02)
        self.encoder = Block(dim, heads=2)
        self.to_decoder = nn.Linear(dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_position = nn.Parameter(torch.randn(1, self.patches, decoder_dim) * 0.02)
        self.decoder = Block(decoder_dim, heads=2)
        self.to_pixels = nn.Linear(decoder_dim, pixels)

    def patchify(self, image):
        batch, channels, height, width = image.shape
        p = self.patch
        x = image.reshape(batch, channels, height // p, p, width // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # axes: batch, patch_rows, patch_columns, rows, columns, channels
        return x.reshape(batch, self.patches, p * p * channels)  # axes: batch, patches, pixels

    def forward(self, image):
        patches = self.patchify(image)  # axes: batch, patches, pixels
        tokens = self.embed(patches) + self.position  # axes: batch, patches, features
        visible = tokens[:, self.visible]  # axes: batch, visible, features
        encoded = self.to_decoder(self.encoder(visible))  # axes: batch, visible, features
        full = self.mask_token.expand(patches.shape[0], self.patches, -1).clone()
        full[:, self.visible] = encoded  # visible tokens return to their places
        decoded = self.decoder(full + self.decoder_position)  # axes: batch, patches, features
        reconstruction = self.to_pixels(decoded)  # axes: batch, patches, pixels
        error = ((reconstruction - patches) ** 2).mean(dim=-1)  # axes: batch, patches
        loss = error[:, self.hidden].mean()
        return reconstruction, loss

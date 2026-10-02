"""CLIP dual encoder.

Two transformers, one for images and one for text, map into a single shared
space. After normalizing both embeddings, the dot product of every image with
the sentence is its matching score, and softmax over the images picks the
best match.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

# track: Multimodal
# input image: batch=2, channels=3, height=16, width=16 | image
# input tokens: text "a photo of a bright disc on a gradient"


class Block(nn.Module):
    def __init__(self, dim, heads, causal=False):
        super().__init__()
        self.heads, self.causal = heads, causal
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
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # axes: batch, heads, queries, keys
        if self.causal:
            allowed = torch.ones(tokens, tokens).tril().bool()  # axes: queries, keys
            scores = scores.masked_fill(~allowed, float("-inf"))
        mixed = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, tokens, dim)  # axes: batch, tokens, features
        x = x + self.output(mixed)
        return x + self.mlp(self.norm2(x))


class ImageEncoder(nn.Module):
    def __init__(self, image_size, patch, channels, dim, embed):
        super().__init__()
        self.patch_embedding = nn.Conv2d(channels, dim, kernel_size=patch, stride=patch)
        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.randn(1, (image_size // patch) ** 2 + 1, dim) * 0.02)
        self.block = Block(dim, heads=2)
        self.norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, embed, bias=False)

    def forward(self, image):
        x = self.patch_embedding(image).flatten(2).transpose(1, 2)  # axes: batch, patches, features
        x = torch.cat([self.class_token.expand(x.shape[0], -1, -1), x], dim=1)  # axes: batch, tokens, features
        x = self.block(x + self.position)
        return self.projection(self.norm(x[:, 0]))  # axes: images, embedding


class TextEncoder(nn.Module):
    def __init__(self, vocabulary, dim, embed, max_length):
        super().__init__()
        self.token = nn.Embedding(vocabulary, dim)
        self.position = nn.Embedding(max_length, dim)
        self.block = Block(dim, heads=2, causal=True)
        self.norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, embed, bias=False)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1]).unsqueeze(0)  # axes: batch, tokens
        x = self.block(self.token(tokens) + self.position(positions))  # axes: batch, tokens, features
        # The last token has attended to the whole sentence.
        return self.projection(self.norm(x[:, -1]))  # axes: sentences, embedding


class CLIP(nn.Module):
    def __init__(self, image_size=16, patch=4, channels=3, vocabulary=64, dim=16, embed=8, max_length=64):
        super().__init__()
        self.image_encoder = ImageEncoder(image_size, patch, channels, dim, embed)
        self.text_encoder = TextEncoder(vocabulary, dim, embed, max_length)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, image, tokens):
        images = F.normalize(self.image_encoder(image), dim=-1)  # axes: images, embedding
        sentences = F.normalize(self.text_encoder(tokens), dim=-1)  # axes: sentences, embedding
        scores = self.log_scale.exp() * images @ sentences.T  # axes: images, sentences
        return scores.softmax(dim=0)  # which image matches the sentence

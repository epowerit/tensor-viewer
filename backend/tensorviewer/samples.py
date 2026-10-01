"""Built-in inputs whose values mean something: a small picture and a tokenized sentence."""

import re
from math import prod

import torch

TOKEN = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']")
MAX_TOKENS = 64


def tokenize(text: str) -> tuple[list[str], list[str]]:
    """Lower-cased words and punctuation, with a sorted vocabulary of the sentence."""
    tokens = [token.lower() for token in TOKEN.findall(text)]
    return tokens, sorted(set(tokens))


def token_ids(text: str) -> list[int]:
    tokens, vocabulary = tokenize(text)
    return [vocabulary.index(token) for token in tokens]


def sample_image(shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    """A deterministic picture: a bright disc on a gradient, in [0, 1].

    The last two axes are height and width. The axis before them selects a
    color channel, and earlier axes move the disc so batch items differ.
    """
    height, width = shape[-2:]
    leading = shape[:-2]
    channels = leading[-1] if leading else 1
    items = prod(leading[:-1])
    y = torch.linspace(0, 1, height, dtype=torch.float64).reshape(height, 1)
    x = torch.linspace(0, 1, width, dtype=torch.float64).reshape(1, width)
    item = torch.arange(items).reshape(items, 1, 1, 1)
    channel = torch.arange(channels).reshape(1, channels, 1, 1) % 3
    center = 0.3 + 0.2 * (item % 3)
    disc = (((x - center) ** 2 + (y - 0.45) ** 2) < 0.075).to(torch.float64)
    red = 0.15 + 0.85 * disc
    green = 0.2 + 0.5 * y + 0.3 * disc
    blue = (0.25 + 0.6 * x) * (1 - 0.7 * disc)
    image = torch.where(channel == 0, red, torch.where(channel == 1, green, blue))
    return image.reshape(shape).to(dtype)

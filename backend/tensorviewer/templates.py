from .models import InputSpec, ProjectDraft, Template

ATTENTION_CODE = '''import math
import torch
from torch import nn


class Attention(nn.Module):
    """Small, explicit attention. Axis comments teach the viewer what dimensions mean."""

    def __init__(self, embed_dim=8, num_heads=2):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        batch, tokens, features = x.shape
        q = self.query(x)  # axes: batch, tokens, features
        k = self.key(x)  # axes: batch, tokens, features
        v = self.value(x)  # axes: batch, tokens, features

        q = q.reshape(batch, tokens, self.num_heads, self.head_dim)  # axes: batch, tokens, heads, head_features
        q = q.permute(0, 2, 1, 3)  # axes: batch, heads, tokens, head_features
        k = k.reshape(batch, tokens, self.num_heads, self.head_dim)  # axes: batch, tokens, heads, head_features
        k = k.permute(0, 2, 1, 3)  # axes: batch, heads, tokens, head_features
        v = v.reshape(batch, tokens, self.num_heads, self.head_dim)  # axes: batch, tokens, heads, head_features
        v = v.permute(0, 2, 1, 3)  # axes: batch, heads, tokens, head_features

        kt = k.transpose(-2, -1)  # axes: batch, heads, head_features, key_tokens
        scores = q @ kt  # axes: batch, heads, query_tokens, key_tokens
        scores = scores / math.sqrt(self.head_dim)  # axes: batch, heads, query_tokens, key_tokens
        weights = torch.softmax(scores, dim=-1)  # axes: batch, heads, query_tokens, key_tokens
        attended = weights @ v  # axes: batch, heads, tokens, head_features

        attended = attended.permute(0, 2, 1, 3)  # axes: batch, tokens, heads, head_features
        attended = attended.contiguous()  # axes: batch, tokens, heads, head_features
        joined = attended.reshape(batch, tokens, features)  # axes: batch, tokens, features
        output = self.projection(joined)  # axes: batch, tokens, features
        return output
'''

BASICS_CODE = '''import torch
from torch import nn


class TensorBasics(nn.Module):
    def __init__(self, groups=2):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch, tokens, features = x.shape
        grouped = x.reshape(batch, tokens, self.groups, features // self.groups)  # axes: batch, tokens, groups, group_features
        reordered = grouped.permute(0, 2, 1, 3)  # axes: batch, groups, tokens, group_features
        packed = reordered.contiguous()  # axes: batch, groups, tokens, group_features
        return packed
'''

TEMPLATES = [
    Template(
        id="attention",
        description="Follow tokens through two attention heads, one operation at a time.",
        project=ProjectDraft(name="Inside attention", code=ATTENTION_CODE),
    ),
    Template(
        id="tensor-basics",
        description="Track numbered elements through reshape, permute, and contiguous.",
        project=ProjectDraft(
            name="A change of shape", code=BASICS_CODE, class_name="TensorBasics",
            constructor={"groups": 2}, input=InputSpec(shape=[1, 2, 8]),
        ),
    ),
]

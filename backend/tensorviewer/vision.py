"""Composable, explicit ViT source included in the generated project for inspection."""

VISION_CODE = '''

class TokenPreparation(nn.Module):
    """Prepend a learned class token, then add learned positions to every token."""
    def __init__(self, tokens, features):
        super().__init__()
        self.tokens = tokens
        self.features = features
        self.class_token = nn.Parameter(torch.zeros(1, 1, features))
        self.positions = nn.Parameter(torch.zeros(1, tokens + 1, features))
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.positions, std=0.02)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.tokens, self.features):
            raise ValueError("Token preparation needs its configured [batch, tokens, features].")
        cls = self.class_token.expand(x.shape[0], -1, -1)  # axes: batch, class_token, features
        joined = torch.cat((cls, x), dim=1)  # axes: batch, tokens, features
        positioned = joined + self.positions  # axes: batch, tokens, features
        return positioned


class ClassTokenReadout(nn.Module):
    """Normalize the sequence and classify the first (class) token."""
    def __init__(self, features, classes):
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.classifier = nn.Linear(features, classes)

    def forward(self, x):
        normalized = self.norm(x)  # axes: batch, tokens, features
        cls = normalized[:, 0, :]  # axes: batch, features
        logits = self.classifier(cls)  # axes: batch, classes
        return logits


class VisionTransformer(nn.Module):
    """A small, untrained ViT with explicit attention and a linear classification head."""
    def __init__(self, channels, height, width, patch=4, features=8, heads=2, blocks=1, expansion=2, classes=10):
        super().__init__()
        if min(channels, height, width, patch, features, heads, blocks, expansion, classes) < 1:
            raise ValueError("ViT dimensions must be positive.")
        if height % patch or width % patch:
            raise ValueError("Image height and width must be divisible by the patch size.")
        if features % heads:
            raise ValueError("Embedding features must divide evenly into attention heads.")
        self.spatial = (height, width)
        self.channels = channels
        self.patches = PatchEmbedding(channels, features, patch)
        self.tokens = TokenPreparation((height // patch) * (width // patch), features)
        self.encoder = nn.Sequential(*[
            TransformerBlock(features, heads, expansion) for _ in range(blocks)
        ])
        self.readout = ClassTokenReadout(features, classes)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels or x.shape[2:] != self.spatial:
            raise ValueError("This ViT expects its configured [batch, channels, height, width].")
        patches = self.patches(x)  # axes: batch, patches, features
        positioned = self.tokens(patches)  # axes: batch, tokens, features
        encoded = self.encoder(positioned)  # axes: batch, tokens, features
        logits = self.readout(encoded)  # axes: batch, classes
        return logits
'''


def transformer_parameters(features: int, blocks: int, expansion: int) -> int:
    """Attention projections, two LayerNorms, and a two-layer biased MLP."""
    return blocks * ((4 + 2 * expansion) * features**2 + (5 + expansion) * features)


def vision_budget(shape, patch, features, heads, blocks, expansion, classes):
    batch, channels, height, width = shape
    tokens = (height // patch) * (width // patch) + 1
    parameters = (
        (channels * patch**2 + 1) * features
        + (tokens + 1) * features
        + transformer_parameters(features, blocks, expansion)
        + 2 * features
        + (features + 1) * classes
    )
    intermediate = max(batch * tokens * features * expansion, batch * heads * tokens**2)
    return parameters, intermediate

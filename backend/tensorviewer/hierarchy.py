"""Explicit spatial building blocks for a small, fixed-window vision hierarchy."""

from .vision import transformer_parameters

HIERARCHY_KINDS = {
    "hierarchical_vit",
    "spatial_embedding",
    "window_partition",
    "window_reverse",
    "window_attention",
    "patch_merging",
    "spatial_readout",
}


HIERARCHY_CODE = '''

class SpatialPatchEmbedding(nn.Module):
    """Project image patches while retaining a [batch, features, height, width] grid."""
    def __init__(self, channels, features=8, patch=2):
        super().__init__()
        self.patch = patch
        self.projection = nn.Conv2d(channels, features, patch, stride=patch)

    def forward(self, x):
        if x.ndim != 4 or x.shape[2] % self.patch or x.shape[3] % self.patch:
            raise ValueError("Spatial patch embedding needs an image divisible by its patch size.")
        grid = self.projection(x)  # axes: batch, features, height, width
        return grid


class WindowPartition(nn.Module):
    """Pack independent square windows; window-batch = batch * windows_per_image + window."""
    def __init__(self, window=2):
        super().__init__()
        if window < 1:
            raise ValueError("Window size must be positive.")
        self.window = window

    def forward(self, x):
        if x.ndim != 4 or x.shape[2] % self.window or x.shape[3] % self.window:
            raise ValueError("Window partition needs [batch, features, height, width] divisible by the window size.")
        batch, channels, height, width = x.shape
        size = self.window
        spatial = x.permute(0, 2, 3, 1)  # axes: batch, height, width, features
        split = spatial.reshape(batch, height // size, size, width // size, size, channels)  # axes: batch, window_rows, local_rows, window_columns, local_columns, features
        grouped = split.permute(0, 1, 3, 2, 4, 5)  # axes: batch, window_rows, window_columns, local_rows, local_columns, features
        windows = grouped.reshape(batch * (height // size) * (width // size), size * size, channels)  # axes: batch_windows, window_tokens, features
        return windows


class WindowReverse(nn.Module):
    """Restore the configured spatial grid from windows in row-major order."""
    def __init__(self, height, width, window=2):
        super().__init__()
        if min(height, width, window) < 1 or height % window or width % window:
            raise ValueError("Output height and width must be divisible by the window size.")
        self.height, self.width, self.window = height, width, window

    def forward(self, x):
        height, width, size = self.height, self.width, self.window
        count = (height // size) * (width // size)
        if x.ndim != 3 or x.shape[1] != size * size or x.shape[0] % count:
            raise ValueError("Window reverse needs [batch * windows_per_image, window_size squared, features].")
        batch, channels = x.shape[0] // count, x.shape[2]
        grouped = x.reshape(batch, height // size, width // size, size, size, channels)  # axes: batch, window_rows, window_columns, local_rows, local_columns, features
        split = grouped.permute(0, 1, 3, 2, 4, 5)  # axes: batch, window_rows, local_rows, window_columns, local_columns, features
        spatial = split.reshape(batch, height, width, channels)  # axes: batch, height, width, features
        grid = spatial.permute(0, 3, 1, 2)  # axes: batch, features, height, width
        return grid


class WindowTransformer(nn.Module):
    """One shared transformer block applied independently to each fixed window."""
    def __init__(self, features, height, width, window=2, heads=1, expansion=2):
        super().__init__()
        if min(features, height, width, window, heads, expansion) < 1 or features % heads:
            raise ValueError("Positive dimensions and evenly divided attention heads are required.")
        self.expected = (features, height, width)
        self.partition = WindowPartition(window)
        self.block = TransformerBlock(features, heads, expansion)
        self.restore = WindowReverse(height, width, window)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != self.expected:
            raise ValueError("This window block needs its configured [batch, features, height, width].")
        windows = self.partition(x)  # axes: batch_windows, window_tokens, features
        attended = self.block(windows)  # axes: batch_windows, window_tokens, features
        grid = self.restore(attended)  # axes: batch, features, height, width
        return grid


class PatchGrouping(nn.Module):
    """Join 2x2 neighbors in top-left, bottom-left, top-right, bottom-right order."""
    def forward(self, x):
        if x.ndim != 4 or x.shape[2] % 2 or x.shape[3] % 2:
            raise ValueError("Patch merging needs even spatial dimensions.")
        batch, channels, height, width = x.shape
        spatial = x.permute(0, 2, 3, 1)  # axes: batch, height, width, features
        split = spatial.reshape(batch, height // 2, 2, width // 2, 2, channels)  # axes: batch, merged_rows, local_rows, merged_columns, local_columns, features
        neighbors = split.permute(0, 1, 3, 4, 2, 5)  # axes: batch, merged_rows, merged_columns, local_columns, local_rows, features
        grouped = neighbors.reshape(batch, height // 2, width // 2, 4 * channels)  # axes: batch, merged_rows, merged_columns, neighbor_features
        return grouped


class PatchMerging(nn.Module):
    """2x2 grouping -> LayerNorm(4C) -> learned 4C-to-2C projection."""
    def __init__(self, features):
        super().__init__()
        self.group = PatchGrouping()
        self.norm = nn.LayerNorm(4 * features)
        self.reduction = nn.Linear(4 * features, 2 * features, bias=False)

    def forward(self, x):
        grouped = self.group(x)  # axes: batch, merged_rows, merged_columns, neighbor_features
        normalized = self.norm(grouped)  # axes: batch, merged_rows, merged_columns, neighbor_features
        reduced = self.reduction(normalized)  # axes: batch, height, width, features
        grid = reduced.permute(0, 3, 1, 2)  # axes: batch, features, height, width
        return grid


class SpatialReadout(nn.Module):
    """Normalize each spatial vector, average over the grid, and produce class logits."""
    def __init__(self, features, classes=10):
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.classifier = nn.Linear(features, classes)

    def forward(self, x):
        spatial = x.permute(0, 2, 3, 1)  # axes: batch, height, width, features
        normalized = self.norm(spatial)  # axes: batch, height, width, features
        pooled = normalized.mean(dim=(1, 2))  # axes: batch, features
        logits = self.classifier(pooled)  # axes: batch, classes
        return logits


class HierarchicalVisionTransformer(nn.Module):
    """Two untrained fixed-window stages; no shifted windows or positional bias."""
    def __init__(self, channels, height, width, patch=2, features=8, window=2, heads=2, expansion=2, classes=10):
        super().__init__()
        if min(channels, height, width, patch, features, window, heads, expansion, classes) < 1:
            raise ValueError("Hierarchy dimensions must be positive.")
        if height % (patch * 2 * window) or width % (patch * 2 * window):
            raise ValueError("Image dimensions must divide into patches, 2x2 merging, and windows at both stages.")
        self.expected = (channels, height, width)
        rows, columns = height // patch, width // patch
        self.embedding = SpatialPatchEmbedding(channels, features, patch)
        self.fine = WindowTransformer(features, rows, columns, window, heads, expansion)
        self.merge = PatchMerging(features)
        self.coarse = WindowTransformer(2 * features, rows // 2, columns // 2, window, heads, expansion)
        self.readout = SpatialReadout(2 * features, classes)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != self.expected:
            raise ValueError("This hierarchy needs its configured image shape; batch size can vary.")
        patches = self.embedding(x)  # axes: batch, features, height, width
        fine = self.fine(patches)  # axes: batch, features, height, width
        merged = self.merge(fine)  # axes: batch, features, height, width
        coarse = self.coarse(merged)  # axes: batch, features, height, width
        logits = self.readout(coarse)  # axes: batch, classes
        return logits
'''


def window_budget(shape, window, heads, expansion):
    batch, features, height, width = shape
    parameters = transformer_parameters(features, 1, expansion)
    # Every window attends only to its own window**2 tokens.
    scores = batch * (height // window) * (width // window) * heads * window**4
    return parameters, max(batch * features * height * width * expansion, scores)


def merge_parameters(features):
    return 8 * features + 8 * features**2


def hierarchy_budget(shape, patch, features, window, heads, expansion, classes):
    batch, channels, height, width = shape
    rows, columns = height // patch, width // patch
    first, first_size = window_budget([batch, features, rows, columns], window, heads, expansion)
    second, second_size = window_budget(
        [batch, features * 2, rows // 2, columns // 2], window, heads, expansion
    )
    parameters = (channels * patch**2 + 1) * features + first + merge_parameters(features) + second
    parameters += 4 * features + (2 * features + 1) * classes
    return parameters, max(first_size, second_size)


def compose_spatial(kind, shape, p, dtype):
    """Shape and allocation rules for the reusable spatial component interfaces."""
    if kind == "window_reverse":
        height, width, window = p["height"], p["width"], p["window"]
        if height % window or width % window:
            raise ValueError("Output height and width must be divisible by the window size.")
        count = (height // window) * (width // window)
        if len(shape) != 3 or shape[1] != window**2 or shape[0] % count:
            raise ValueError(
                "Window reverse needs [batch × windows per image, window size², features]."
            )
        return (
            [shape[0] // count, shape[2], height, width],
            ["batch", "features", "height", "width"],
            f"WindowReverse({height}, {width}, {window})",
            0,
            0,
        )
    if len(shape) != 4:
        raise ValueError("Needs [batch, channels/features, height, width]. Choose an image input.")
    if kind != "window_partition" and dtype == "int64":
        raise ValueError("This learned spatial component needs floating-point input.")
    batch, channels, height, width = shape
    axes = ["batch", "features", "height", "width"]
    parameters, intermediate = 0, 0
    if kind == "hierarchical_vit":
        factor = p["patch"] * 2 * p["window"]
        if height % factor or width % factor:
            raise ValueError(
                f"Image height and width must be divisible by {factor} (patch × 2 × window), so windows fit both stages."
            )
        if p["features"] % p["heads"]:
            raise ValueError("Embedding features must divide evenly into attention heads.")
        parameters, intermediate = hierarchy_budget(shape, **p)
        module = f"HierarchicalVisionTransformer({channels}, {height}, {width}, patch={p['patch']}, features={p['features']}, window={p['window']}, heads={p['heads']}, expansion={p['expansion']}, classes={p['classes']})"
        shape, axes = [batch, p["classes"]], ["batch", "classes"]
    elif kind == "spatial_embedding":
        if height % p["patch"] or width % p["patch"]:
            raise ValueError("Image height and width must be divisible by the patch size.")
        module = f"SpatialPatchEmbedding({channels}, {p['features']}, {p['patch']})"
        parameters = (channels * p["patch"] ** 2 + 1) * p["features"]
        shape = [batch, p["features"], height // p["patch"], width // p["patch"]]
    elif kind in {"window_partition", "window_attention"}:
        size = p["window"]
        if height % size or width % size:
            raise ValueError(
                "Height and width must be divisible by the window size; no padding is added."
            )
        if kind == "window_partition":
            module = f"WindowPartition({size})"
            shape = [batch * (height // size) * (width // size), size**2, channels]
            axes = ["batch_windows", "window_tokens", "features"]
        else:
            if channels % p["heads"]:
                raise ValueError("Input features must divide evenly into attention heads.")
            parameters, intermediate = window_budget(shape, **p)
            module = f"WindowTransformer({channels}, {height}, {width}, {size}, {p['heads']}, {p['expansion']})"
    elif kind == "patch_merging":
        if height % 2 or width % 2:
            raise ValueError("Patch merging needs even height and width; no padding is added.")
        module = f"PatchMerging({channels})"
        parameters = merge_parameters(channels)
        shape = [batch, 2 * channels, height // 2, width // 2]
    else:
        module = f"SpatialReadout({channels}, {p['classes']})"
        parameters = 2 * channels + (channels + 1) * p["classes"]
        shape, axes = [batch, p["classes"]], ["batch", "classes"]
    return shape, axes, module, parameters, intermediate

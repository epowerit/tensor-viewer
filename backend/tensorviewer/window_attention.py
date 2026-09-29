"""Inspectable relative-position attention and cyclically shifted spatial windows."""

from .hierarchy import window_budget

WINDOW_KINDS = {"shifted_window", "window_pair"}

WINDOW_CODE = '''

class WindowScoreAdjustment(nn.Module):
    """Learn relative offsets and mask cyclic wraparound before row-wise softmax."""
    def __init__(self, height, width, window, heads, shift_y=0, shift_x=0):
        super().__init__()
        self.windows = (height // window) * (width // window)
        self.relative_bias_table = nn.Parameter(torch.empty((2 * window - 1) ** 2, heads))
        nn.init.normal_(self.relative_bias_table, std=0.02)
        positions = torch.arange(window * window)
        rows, columns = positions // window, positions % window
        relative_index = (rows[:, None] - rows[None, :] + window - 1) * (2 * window - 1)
        relative_index = relative_index + columns[:, None] - columns[None, :] + window - 1
        self.register_buffer("relative_index", relative_index)
        # Only the last window along a shifted axis contains wrapped positions.
        y = torch.arange(height) >= height - shift_y
        x = torch.arange(width) >= width - shift_x
        regions = y[:, None].to(torch.int64) * 2 + x[None, :].to(torch.int64)
        regions = regions.reshape(height // window, window, width // window, window)
        regions = regions.permute(0, 2, 1, 3).reshape(self.windows, window * window)
        blocked = (regions[:, :, None] != regions[:, None, :]).unsqueeze(1)
        self.register_buffer("blocked", blocked)

    def forward(self, scores):
        count, heads, tokens, _ = scores.shape
        relative = self.relative_bias_table[self.relative_index]  # axes: query_tokens, key_tokens, heads
        bias = relative.permute(2, 0, 1)  # axes: heads, query_tokens, key_tokens
        positioned = scores + bias  # axes: batch_windows, heads, query_tokens, key_tokens
        grouped = positioned.reshape(count // self.windows, self.windows, heads, tokens, tokens)  # axes: batch, windows, heads, query_tokens, key_tokens
        masked = grouped.masked_fill(self.blocked, float("-inf"))  # axes: batch, windows, heads, query_tokens, key_tokens
        flat = masked.reshape(count, heads, tokens, tokens)  # axes: batch_windows, heads, query_tokens, key_tokens
        weights = torch.softmax(flat, dim=-1)  # axes: batch_windows, heads, query_tokens, key_tokens
        return weights


class RelativeWindowAttention(Attention):
    """Explicit Q/K/V with a learned bias per local query-key offset and head."""
    def __init__(self, features, height, width, window, heads, shift_y, shift_x):
        super().__init__(features, heads)
        self.adjust = WindowScoreAdjustment(height, width, window, heads, shift_y, shift_x)

    def forward(self, x):
        count, tokens, features = x.shape
        q = self.query(x)  # axes: batch_windows, window_tokens, features
        k = self.key(x)  # axes: batch_windows, window_tokens, features
        v = self.value(x)  # axes: batch_windows, window_tokens, features
        q = q.reshape(count, tokens, self.num_heads, self.head_dim)  # axes: batch_windows, window_tokens, heads, head_features
        q = q.permute(0, 2, 1, 3)  # axes: batch_windows, heads, query_tokens, head_features
        k = k.reshape(count, tokens, self.num_heads, self.head_dim)  # axes: batch_windows, window_tokens, heads, head_features
        k = k.permute(0, 2, 1, 3)  # axes: batch_windows, heads, key_tokens, head_features
        v = v.reshape(count, tokens, self.num_heads, self.head_dim)  # axes: batch_windows, window_tokens, heads, head_features
        v = v.permute(0, 2, 1, 3)  # axes: batch_windows, heads, key_tokens, head_features
        kt = k.transpose(-2, -1)  # axes: batch_windows, heads, head_features, key_tokens
        scores = q @ kt  # axes: batch_windows, heads, query_tokens, key_tokens
        scores = scores / math.sqrt(self.head_dim)  # axes: batch_windows, heads, query_tokens, key_tokens
        weights = self.adjust(scores)  # axes: batch_windows, heads, query_tokens, key_tokens
        attended = weights @ v  # axes: batch_windows, heads, window_tokens, head_features
        attended = attended.permute(0, 2, 1, 3)  # axes: batch_windows, window_tokens, heads, head_features
        joined = attended.reshape(count, tokens, features)  # axes: batch_windows, window_tokens, features
        output = self.projection(joined)  # axes: batch_windows, window_tokens, features
        return output


class ShiftedWindowTransformer(nn.Module):
    """Shift -> partition -> pre-norm attention/MLP -> restore -> undo shift."""
    def __init__(self, features, height, width, window=2, shift=1, heads=1, expansion=2):
        super().__init__()
        if min(features, height, width, window, heads, expansion) < 1 or features % heads:
            raise ValueError("Positive dimensions and evenly divided heads are required.")
        if height % window or width % window or not 0 <= shift < window:
            raise ValueError("Window size must divide the grid; shift must be between 0 and window size minus one.")
        self.expected = (features, height, width)
        self.shift_y = shift if height > window else 0
        self.shift_x = shift if width > window else 0
        self.partition = WindowPartition(window)
        self.norm1 = nn.LayerNorm(features)
        self.attention = RelativeWindowAttention(features, height, width, window, heads, self.shift_y, self.shift_x)
        self.norm2 = nn.LayerNorm(features)
        self.feedforward = nn.Sequential(nn.Linear(features, features * expansion), nn.GELU(), nn.Linear(features * expansion, features))
        self.restore = WindowReverse(height, width, window)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != self.expected:
            raise ValueError("This shifted-window block needs its configured spatial shape; batch size can vary.")
        shifted = torch.roll(x, shifts=(-self.shift_y, -self.shift_x), dims=(2, 3))  # axes: batch, features, height, width
        windows = self.partition(shifted)  # axes: batch_windows, window_tokens, features
        normalized = self.norm1(windows)  # axes: batch_windows, window_tokens, features
        attended = self.attention(normalized)  # axes: batch_windows, window_tokens, features
        residual = windows + attended  # axes: batch_windows, window_tokens, features
        normalized = self.norm2(residual)  # axes: batch_windows, window_tokens, features
        features = self.feedforward(normalized)  # axes: batch_windows, window_tokens, features
        combined = residual + features  # axes: batch_windows, window_tokens, features
        grid = self.restore(combined)  # axes: batch, features, height, width
        output = torch.roll(grid, shifts=(self.shift_y, self.shift_x), dims=(2, 3))  # axes: batch, features, height, width
        return output


class AlternatingWindowPair(nn.Module):
    """Regular windows then shifted windows, with separate learned weights and biases."""
    def __init__(self, features, height, width, window=2, shift=1, heads=1, expansion=2):
        super().__init__()
        self.regular = ShiftedWindowTransformer(features, height, width, window, 0, heads, expansion)
        self.shifted = ShiftedWindowTransformer(features, height, width, window, shift, heads, expansion)

    def forward(self, x):
        regular = self.regular(x)  # axes: batch, features, height, width
        shifted = self.shifted(regular)  # axes: batch, features, height, width
        return shifted
'''


def compose_window(kind, shape, p, dtype):
    if len(shape) != 4:
        raise ValueError("Needs [batch, features, height, width]. Choose a spatial input.")
    if dtype == "int64":
        raise ValueError("Window transformers need floating-point input.")
    _, features, height, width = shape
    if height % p["window"] or width % p["window"]:
        raise ValueError(
            "Height and width must be divisible by the window size; no padding is added."
        )
    if not 0 <= p["shift"] < p["window"]:
        raise ValueError("Shift must be between 0 and window size minus one.")
    if features % p["heads"]:
        raise ValueError("Features must divide evenly into attention heads.")
    parameters, peak = window_budget(shape, p["window"], p["heads"], p["expansion"])
    parameters += (2 * p["window"] - 1) ** 2 * p["heads"]
    module = "AlternatingWindowPair" if kind == "window_pair" else "ShiftedWindowTransformer"
    if kind == "window_pair":
        parameters *= 2
    source = f"{module}({features}, {height}, {width}, {p['window']}, {p['shift']}, {p['heads']}, {p['expansion']})"
    return shape, ["batch", "features", "height", "width"], source, parameters, peak

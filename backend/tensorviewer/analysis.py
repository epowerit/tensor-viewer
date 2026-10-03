"""Whole-tensor questions answered where the values live.

Tensors too large to send inline stay in snapshots; the browser asks these
functions instead: where values match a test, where the extremes sit, row and
column reductions of a plane, and totals over a selected rectangle.
"""

import io

import numpy as np

from .snapshots import json_value

TESTS = {"gt", "ge", "lt", "le", "eq", "ne", "between", "nan", "inf", "finite"}
REDUCTIONS = {"sum", "mean", "max", "min"}
#: Larger planes are not reduced; their margins would not fit a response.
PLANE_LIMIT = 4_000_000


def as_array(values) -> np.ndarray:
    """Inline JSON values as floats; Python's 'nan', 'inf' and '-inf' included."""
    return np.array([float(v) for v in values], dtype=np.float64)


def matches(flat: np.ndarray, test: str, value=0.0, high=0.0, magnitude=False, closed=False):
    """A boolean mask of the values passing `test`; comparisons skip NaN and ±inf."""
    data = flat.astype(np.float64, copy=False)
    if test == "nan":
        return np.isnan(data)
    if test == "inf":
        return np.isinf(data)
    finite = np.isfinite(data)
    if test == "finite":
        return finite
    x = np.abs(data) if magnitude else data
    with np.errstate(invalid="ignore"):
        if test == "between":
            # Half-open like a histogram bar; `closed` keeps the last bar's end.
            return finite & (x >= value) & ((x < high) | (closed & (x == high)))
        return (
            finite
            & {
                "gt": x > value,
                "ge": x >= value,
                "lt": x < value,
                "le": x <= value,
                "eq": x == value,
                "ne": x != value,
            }[test]
        )


def search(
    flat: np.ndarray, test: str, value=0.0, high=0.0, magnitude=False, closed=False, limit=5000
):
    """How many values pass, and the first `limit` of their flat indices."""
    found = np.flatnonzero(matches(flat, test, value, high, magnitude, closed))
    return {
        "count": int(found.size),
        "indices": found[:limit].tolist(),
        "truncated": bool(found.size > limit),
    }


def landmarks(flat: np.ndarray, limit=5000):
    """The first smallest and largest finite values, and every non-finite one."""
    data = flat.astype(np.float64, copy=False)
    finite = np.isfinite(data)
    broken = np.flatnonzero(~finite)
    if finite.any():
        masked_low = np.where(finite, data, np.inf)
        masked_high = np.where(finite, data, -np.inf)
        low, high = int(np.argmin(masked_low)), int(np.argmax(masked_high))
    else:
        low = high = None
    values = data[finite]
    return {
        "min": low,
        "max": high,
        "broken": broken[:limit].tolist(),
        "broken_count": int(broken.size),
        # Linear interpolation, as numpy.percentile and the browser compute.
        "quantiles": (
            [json_value(q) for q in np.percentile(values, [1, 50, 99])] if values.size else None
        ),
    }


def plane(array: np.ndarray, row: int | None, column: int | None, fixed: list[int]):
    """The 2-D plane through `fixed`, rows by columns."""
    index = tuple(
        slice(None) if axis in (row, column) else fixed[axis] for axis in range(array.ndim)
    )
    view = array[index]
    kept = [axis for axis in range(array.ndim) if axis in (row, column)]
    # Indexing keeps the remaining axes in their original order.
    if row is not None and column is not None and kept.index(row) > kept.index(column):
        view = view.T
    if row is None:
        view = view.reshape(1, -1)
    if column is None:
        view = view.reshape(-1, 1)
    return np.asarray(view, dtype=np.float64)


def reduce(values: np.ndarray, how: str, axis=None):
    # NaN spreads, as it does in torch.sum, mean, max and min.
    return {"sum": np.sum, "mean": np.mean, "max": np.max, "min": np.min}[how](values, axis=axis)


def margins(array: np.ndarray, row, column, fixed, how: str):
    grid = plane(array, row, column, fixed)
    if grid.size > PLANE_LIMIT:
        raise ValueError("This plane is too large to reduce.")
    return {
        "rows": [json_value(v) for v in reduce(grid, how, axis=1)],
        "columns": [json_value(v) for v in reduce(grid, how, axis=0)],
        "all": json_value(reduce(grid, how)),
    }


def region(
    array: np.ndarray,
    row,
    column,
    fixed,
    rows: tuple[int, int],
    columns: tuple[int, int],
    values_limit=10_000,
):
    """Totals over a rectangle of the plane, and its values when few enough."""
    block = plane(array, row, column, fixed)[rows[0] : rows[1] + 1, columns[0] : columns[1] + 1]
    finite = block[np.isfinite(block)]
    stats = (
        {
            "count": int(finite.size),
            "sum": json_value(finite.sum()),
            "mean": json_value(finite.mean()),
            "std": json_value(finite.std()),
            "min": json_value(finite.min()),
            "max": json_value(finite.max()),
        }
        if finite.size
        else {"count": 0, "sum": None, "mean": None, "std": None, "min": None, "max": None}
    )
    return {
        **stats,
        "broken": int(block.size - finite.size),
        "values": [json_value(v) for v in block.reshape(-1)]
        if block.size <= values_limit
        else None,
    }


def compare(now: np.ndarray, before: np.ndarray):
    """How a tensor state changed since an earlier one of the same size.

    Matches the browser's cell deltas: finite pairs compare by difference; a
    NaN or infinity on either side counts as changed unless both sides hold
    the same non-finite value. `low` and `high` bound the finite differences,
    and zero.
    """
    a = now.reshape(-1).astype(np.float64, copy=False)
    b = before.reshape(-1).astype(np.float64, copy=False)
    if a.size != b.size:
        return None
    finite = np.isfinite(a) & np.isfinite(b)
    delta = a[finite] - b[finite]
    faulty = ~finite
    same_fault = (np.isnan(a) & np.isnan(b)) | (np.isinf(a) & (a == b))
    changed = int(np.count_nonzero(delta)) + int(np.count_nonzero(faulty & ~same_fault))
    magnitude = np.abs(delta)
    return {
        "changed": changed,
        "compared": int(a.size),
        "low": json_value(min(0.0, float(delta.min()))) if delta.size else 0.0,
        "high": json_value(max(0.0, float(delta.max()))) if delta.size else 0.0,
        "max_abs": json_value(magnitude.max()) if delta.size else 0.0,
        "mean_abs": json_value(magnitude.mean()) if delta.size else 0.0,
        # torch.allclose with its defaults: NaN is never close, equal infinities are.
        "allclose": bool(np.allclose(a, b, rtol=1e-5, atol=1e-8)),
    }


#: Axes longer than this are not profiled index by index.
PROFILE_LIMIT = 16_384


def axis_profile(array: np.ndarray, axis: int):
    """Mean, σ, min and max of each index along `axis`, over every other axis.

    NaN spreads into the index it sits in, as in torch.
    """
    if array.shape[axis] > PROFILE_LIMIT:
        raise ValueError("This axis is too long to profile.")
    moved = np.moveaxis(np.asarray(array, dtype=np.float64), axis, 0)
    rows = moved.reshape(moved.shape[0], -1)
    return {
        stat: [json_value(v) for v in getattr(np, stat)(rows, axis=1)]
        for stat in ("mean", "std", "min", "max")
    }


#: Thumbnails stop at this many slices, and at this much work in all.
THUMBNAIL_SLICES = 256
THUMBNAIL_CELLS = 50_000_000


def downsample(grid: np.ndarray, size: int) -> np.ndarray:
    """A plane averaged into at most `size` × `size` blocks."""
    rows = np.array_split(np.arange(grid.shape[0]), min(size, grid.shape[0]))
    columns = np.array_split(np.arange(grid.shape[1]), min(size, grid.shape[1]))
    return np.array([[grid[np.ix_(r, c)].mean() for c in columns] for r in rows])


def thumbnails(array: np.ndarray, row, column, fixed: list[int], axis: int, size=8):
    """A small averaged picture of the plane at each index of `axis`.

    The other hidden axes stay at `fixed`. NaN spreads into its block.
    """
    count = min(array.shape[axis], THUMBNAIL_SLICES)
    first = plane(array, row, column, fixed)
    if first.size * count > THUMBNAIL_CELLS:
        raise ValueError("These slices are too large to picture.")
    pictures = []
    for index in range(count):
        coordinates = list(fixed)
        coordinates[axis] = index
        pictures.append(downsample(plane(array, row, column, coordinates), size))
    return {
        "rows": int(pictures[0].shape[0]),
        "columns": int(pictures[0].shape[1]),
        "total": int(array.shape[axis]),
        "thumbs": [[json_value(v) for v in picture.reshape(-1)] for picture in pictures],
    }


def sums(array: np.ndarray, axis: int, value: float):
    """Every sum over `axis` compared with `value`, as a `sums(axis): v`
    contract asks: how many there are, how many are off, and the worst."""
    totals = np.asarray(array, dtype=np.float64).sum(axis=axis).reshape(-1)
    tolerance = 1e-4 * max(1.0, abs(value))
    with np.errstate(invalid="ignore"):
        off = ~(np.abs(totals - value) <= tolerance)
    worst = None
    if off.any():
        bad = totals[off]
        nan = np.isnan(bad)
        worst = json_value(bad[nan][0] if nan.any() else bad[np.argmax(np.abs(bad - value))])
    return {"count": int(totals.size), "off": int(off.sum()), "worst": worst}


NUMPY_DTYPES = {
    "bool": np.bool_,
    "uint8": np.uint8,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}


def npy_bytes(array: np.ndarray, dtype: str) -> bytes:
    """The values as a .npy file; bfloat16 and unknown dtypes become float32."""
    target = NUMPY_DTYPES.get(dtype.removeprefix("torch."), np.float32)
    stream = io.BytesIO()
    np.save(stream, np.asarray(array).astype(target, copy=False), allow_pickle=False)
    return stream.getvalue()

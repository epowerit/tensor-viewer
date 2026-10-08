"""Evaluates one watch expression over a run's recorded tensors, in its own
process. Trusted local code, like a run: this is not a security sandbox."""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Values sent back for a tensor result; the rest are summarized.
SHOWN = 64


def number(value):
    """A recorded value as Python reads it: 'nan' and 'inf' come back as floats."""
    return float(value) if isinstance(value, str) else value


def load(spec: dict) -> torch.Tensor:
    dtype = getattr(torch, spec["dtype"], torch.float32)
    if spec.get("path"):
        array = np.load(spec["path"], allow_pickle=False)
        return torch.from_numpy(np.array(array)).to(dtype)
    values = [number(value) for value in spec["values"]]
    return torch.tensor(values, dtype=dtype).reshape(spec["shape"])


def plain(value):
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def describe(result) -> dict:
    if isinstance(result, torch.Tensor):
        data = result.detach()
        flat = data.reshape(-1)
        described = {
            "kind": "tensor",
            "shape": list(data.shape),
            "dtype": str(data.dtype).removeprefix("torch."),
            "numel": data.numel(),
            "values": [plain(v) for v in flat[:SHOWN].tolist()],
        }
        if data.numel():
            numeric = flat.to(torch.float64)
            finite = numeric[torch.isfinite(numeric)]
            if finite.numel():
                described["stats"] = {
                    "min": float(finite.min()),
                    "max": float(finite.max()),
                    "mean": float(finite.mean()),
                    "std": float(finite.std(unbiased=False)),
                }
            described["non_finite"] = int(numeric.numel() - finite.numel())
        return described
    if isinstance(result, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in result):
        return {
            "kind": "value",
            "text": f"{type(result).__name__} of {len(result)} tensors: "
            + ", ".join(f"[{', '.join(map(str, t.shape))}]" for t in result[:8]),
        }
    return {"kind": "value", "text": repr(plain(result))[:2000]}


def evaluate(request: dict) -> dict:
    names = {name: load(spec) for name, spec in request["tensors"].items()}
    params = {name: load(spec) for name, spec in request.get("params", {}).items()}
    scope = {"torch": torch, "F": F, "math": math, "params": params, **names}
    try:
        code = compile(request["expression"], "<watch>", "eval")
    except SyntaxError as error:
        return {"kind": "error", "text": f"SyntaxError: {error.msg}"}
    try:
        with torch.no_grad():
            return describe(eval(code, scope))
    except Exception as error:
        return {"kind": "error", "text": f"{type(error).__name__}: {error}"}


def single(result) -> float:
    """A result as one number, or why it is not one."""
    if isinstance(result, torch.Tensor):
        if result.numel() != 1:
            raise ValueError(f"not a single number: shape [{', '.join(map(str, result.shape))}]")
        result = result.item()
    if isinstance(result, bool | int | float):
        return float(result)
    raise ValueError(f"not a number: {type(result).__name__}")


def series(request: dict) -> dict:
    """The expression at each of several steps, each a set of named states."""
    states = {key: load(spec) for key, spec in request["states"].items()}
    params = {name: load(spec) for name, spec in request.get("params", {}).items()}
    try:
        code = compile(request["expression"], "<watch>", "eval")
    except SyntaxError as error:
        return {"error": f"SyntaxError: {error.msg}", "points": []}
    points = []
    for point in request["points"]:
        scope = {"torch": torch, "F": F, "math": math, "params": params}
        scope.update({name: states[key] for name, key in point["names"].items()})
        try:
            with torch.no_grad():
                value = single(eval(code, scope))
            points.append({"at": point["at"], "value": value if math.isfinite(value) else None})
        except Exception as error:
            points.append(
                {"at": point["at"], "value": None, "error": f"{type(error).__name__}: {error}"}
            )
    return {"points": points}


def spectra(request: dict) -> dict:
    """Each weight's size and the singular values of it as a matrix.

    A weight of more than two axes is read as [first axis, everything else],
    as a convolution's [out, in·kh·kw]; a vector has a norm but no spectrum.
    """
    found = []
    against = request.get("against", {})
    for name, spec in request["weights"].items():
        weight = load(spec).detach().to(torch.float64)
        entry = {"name": name, "norm": float(weight.norm()), **spectrum(weight)}
        # What changed since the other run: the update and its own spectrum.
        if name in against:
            before = load(against[name]).detach().to(torch.float64)
            if before.shape == weight.shape:
                update = weight - before
                size = float(update.norm())
                entry["update"] = {
                    "norm": size,
                    "relative": size / float(before.norm()) if float(before.norm()) > 0 else None,
                    **spectrum(update),
                }
        found.append(entry)
    return {"weights": found}


def spectrum(weight: torch.Tensor) -> dict:
    """A tensor read as [first axis, everything else]: its singular values,
    numerical rank, effective rank, and condition. A vector has none."""
    if weight.dim() < 2 or not weight.numel():
        return {}
    matrix = weight.reshape(weight.shape[0], -1)
    values = torch.linalg.svdvals(matrix)
    top = float(values[0]) if values.numel() else 0.0
    tolerance = top * max(matrix.shape) * torch.finfo(torch.float32).eps
    share = values / values.sum() if float(values.sum()) > 0 else values
    positive = share[share > 0]
    entropy = float(-(positive * positive.log()).sum())
    return {
        "singular": [float(v) for v in values[:64]],
        "rank": int((values > tolerance).sum()),
        "full": min(matrix.shape),
        "effective_rank": math.exp(entropy) if top > 0 else 0.0,
        "condition": float(values[0] / values[-1]) if float(values[-1]) > 0 else None,
    }


if __name__ == "__main__":
    from .standby import receive

    receive()
    request_path, response_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text())
    result = (
        spectra(request)
        if "weights" in request
        else series(request)
        if "points" in request
        else evaluate(request)
    )
    response_path.write_text(json.dumps(result, allow_nan=False))
    # Answered: leave without tearing down PyTorch (see worker.py).
    os._exit(0)

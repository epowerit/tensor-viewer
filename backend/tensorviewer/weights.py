"""Immutable, bounded tensor-only checkpoints; no unrestricted pickle fallback."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import zipfile
from collections import OrderedDict
from pathlib import Path

import torch

from .input_files import checksum
from .models import SavedWeights, WeightCheck, WeightTensor

MAX_WEIGHT_BYTES = 64 * 1024 * 1024
MAX_TENSORS = 2048
MAX_TENSOR_ELEMENTS = 8_388_608


def validate_archive(path: Path):
    if not 0 < path.stat().st_size <= MAX_WEIGHT_BYTES:
        raise ValueError("Use a checkpoint file up to 64 MiB.")
    if not zipfile.is_zipfile(path):
        raise ValueError(
            "Use the current torch.save state_dict format (.pt or .pth). Legacy pickle files are not supported."
        )
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(entries) > MAX_TENSORS + 16 or len(set(names)) != len(names):
            raise ValueError("The checkpoint contains too many or duplicate archive entries.")
        if sum(entry.file_size for entry in entries) > MAX_WEIGHT_BYTES:
            raise ValueError("The expanded checkpoint exceeds 64 MiB.")
        if any(entry.compress_type != zipfile.ZIP_STORED for entry in entries):
            raise ValueError("Use an uncompressed torch.save checkpoint.")
        pickles = [entry for entry in entries if entry.filename.endswith("/data.pkl")]
        if len(pickles) != 1 or pickles[0].file_size > 1024 * 1024:
            raise ValueError(
                "The checkpoint has an unsupported or oversized state dictionary header."
            )
        if any("/code/" in name or name.endswith("/constants.pkl") for name in names):
            raise ValueError("Export model.state_dict(), not a TorchScript model.")


def read_state(path: Path, *, metadata: bool):
    validate_archive(path)
    try:
        state = torch.load(
            path, map_location="meta" if metadata else "cpu", weights_only=True, mmap=True
        )
    except Exception:
        # Do not echo torch.load's suggestions to disable weights_only or allow custom globals.
        raise ValueError(
            "Cannot read tensor-only weights. Export with torch.save(model.state_dict(), 'weights.pt'). Full models and custom objects are not supported."
        ) from None
    if type(state) not in (dict, OrderedDict):
        raise ValueError("The checkpoint must contain a tensor state dictionary.")
    # Common training-checkpoint wrappers, with no guessing between ambiguous dictionaries.
    wrappers = [
        key
        for key in ("state_dict", "model_state_dict")
        if type(state.get(key)) in (dict, OrderedDict)
    ]
    if len(wrappers) > 1:
        raise ValueError(
            "Export one state dictionary; this checkpoint contains multiple candidates."
        )
    if wrappers:
        state = state[wrappers[0]]
    if not 1 <= len(state) <= MAX_TENSORS:
        raise ValueError("Use a state dictionary with 1–2,048 tensors.")
    total = 0
    for name, value in state.items():
        if not isinstance(name, str) or not name or len(name) > 256:
            raise ValueError("Weight names must be nonempty strings of at most 256 characters.")
        if (
            type(value) is not torch.Tensor
            or value.layout != torch.strided
            or value.is_complex()
            or value.is_quantized
        ):
            raise ValueError(f"{name}: only dense, real tensor weights and buffers are supported.")
        if not metadata and value.device.type != "cpu":
            raise ValueError(f"{name}: export materialized tensor values, not meta tensors.")
        if value.ndim > 8 or value.numel() > MAX_TENSOR_ELEMENTS:
            raise ValueError(
                f"{name}: use at most eight dimensions and 8,388,608 elements per tensor."
            )
        total += value.numel() * value.element_size()
    if total > MAX_WEIGHT_BYTES - 1024 * 1024:
        raise ValueError("The tensor values exceed the checkpoint budget (63 MiB plus metadata).")
    # Standard nn.Module state_dict version metadata is needed by load hooks.
    metadata_dict = getattr(state, "_metadata", {})
    try:
        if len(json.dumps(metadata_dict, allow_nan=False)) > 64_000:
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError(
            "Checkpoint version metadata must be a small JSON-compatible dictionary."
        ) from None
    return state


def tensor_manifest(state):
    return [
        WeightTensor(
            name=name, shape=list(value.shape), dtype=str(value.dtype).removeprefix("torch.")
        )
        for name, value in state.items()
    ]


def normalize_checkpoint(source: Path, destination: Path):
    read_state(
        source, metadata=True
    )  # Validate declared tensors before allocating numeric storage.
    state = read_state(source, metadata=False)
    # Save each tensor's logical values; never keep a view onto the uploaded archive.
    normalized = OrderedDict(
        (name, value.detach().cpu().contiguous().clone()) for name, value in state.items()
    )
    normalized._metadata = getattr(state, "_metadata", {})
    torch.save(normalized, destination)
    validate_archive(destination)
    return tensor_manifest(normalized)


def import_checkpoint(source: Path, destination: Path, timeout: float = 20) -> list[WeightTensor]:
    """Parsing uploads runs outside the web server with a fixed deadline."""
    with tempfile.TemporaryDirectory(prefix="tensorviewer-weights-") as directory:
        response = Path(directory) / "response.json"
        env = {
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tensorviewer.weights",
                str(source.resolve()),
                str(destination.resolve()),
                str(response),
            ],
            cwd=directory,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise ValueError(
                "Checkpoint import exceeded 20 seconds. Use a smaller tensor-only state dictionary."
            ) from None
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        if process.returncode != 0 or not response.is_file() or response.stat().st_size > 1_000_000:
            raise ValueError("The checkpoint importer could not read this file.")
        result = json.loads(response.read_text())
        if "error" in result:
            raise ValueError(result["error"])
        return [WeightTensor.model_validate(item) for item in result["tensors"]]


def load_weights(directory: Path | None, saved: SavedWeights, *, metadata: bool):
    path = directory / f"{saved.id}.pt" if directory is not None else None
    if path is None or not path.is_file():
        raise ValueError(
            "The saved weights file is missing. Import it again or restore the weights backup."
        )
    if path.stat().st_size != saved.byte_count or checksum(path) != saved.sha256:
        raise ValueError(
            "The saved weights file has changed. Import it again to use the new checkpoint."
        )
    state = read_state(path, metadata=metadata)
    if tensor_manifest(state) != saved.tensors:
        raise ValueError("The saved weights metadata does not match the checkpoint.")
    return state


def check_compatibility(model, state) -> WeightCheck:
    expected = model.state_dict()
    issues = []
    for name in expected:
        if name not in state:
            issues.append(f"Missing weight: {name}")
    for name, tensor in state.items():
        if name not in expected:
            issues.append(f"Unexpected weight: {name}")
        elif not isinstance(expected[name], torch.Tensor):
            issues.append(f"{name}: non-tensor module state is not supported.")
        elif tensor.shape != expected[name].shape:
            issues.append(
                f"{name}: checkpoint shape {list(tensor.shape)}; model expects {list(expected[name].shape)}."
            )
        elif tensor.dtype != expected[name].dtype:
            issues.append(
                f"{name}: checkpoint dtype {tensor.dtype}; model expects {expected[name].dtype}. Match the first input's floating dtype or export matching weights."
            )
    return WeightCheck(
        compatible=not issues,
        issues=issues[:24]
        + ([f"And {len(issues) - 24} more mismatches."] if len(issues) > 24 else []),
        tensor_count=len(state),
    )


if __name__ == "__main__":
    source, destination, response = map(Path, sys.argv[1:4])
    try:
        torch.set_num_threads(1)
        result = {
            "tensors": [item.model_dump() for item in normalize_checkpoint(source, destination)]
        }
    except Exception as exc:
        destination.unlink(missing_ok=True)
        result = {"error": str(exc)[:3000]}
    response.write_text(json.dumps(result, allow_nan=False))

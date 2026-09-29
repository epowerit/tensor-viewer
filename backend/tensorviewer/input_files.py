"""Bounded, pickle-free NumPy inputs. Files contain values, never executable code."""

import hashlib
from math import prod
from pathlib import Path

import numpy as np

from .models import UploadedTensor

MAX_ELEMENTS = 8_388_608
MAX_HEADER_BYTES = 16_384
MAX_UPLOAD_BYTES = MAX_ELEMENTS * 8 + MAX_HEADER_BYTES
DTYPES = {"float32", "float64", "int64"}


def inspect_array(path: Path) -> tuple[list[int], np.dtype]:
    """Validate declared allocation and exact payload size before mapping any data."""
    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise ValueError("Use a NumPy file smaller than 64 MiB plus its header.")
    try:
        with path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            if version not in {(1, 0), (2, 0)}:
                raise ValueError("Use NumPy .npy format version 1 or 2.")
            header_start = stream.tell()
            length_bytes = 2 if version == (1, 0) else 4
            header_length = int.from_bytes(stream.read(length_bytes), "little")
            if not 0 < header_length <= MAX_HEADER_BYTES:
                raise ValueError("The NumPy header exceeds the supported size.")
            if header_start + length_bytes + header_length > size:
                raise ValueError("The NumPy file header is truncated.")
            stream.seek(header_start)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(
                    stream, max_header_size=MAX_HEADER_BYTES
                )
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(
                    stream, max_header_size=MAX_HEADER_BYTES
                )
            if dtype.fields or dtype.subdtype or dtype.hasobject or dtype.name not in DTYPES:
                raise ValueError(
                    "Use float32, float64, or int64 values. Object, structured, and other dtypes are not supported."
                )
            if not 1 <= len(shape) <= 6 or any(n < 1 for n in shape) or prod(shape) > MAX_ELEMENTS:
                raise ValueError(
                    "Upload a non-empty rank 1–6 tensor with at most 8,388,608 elements."
                )
            if stream.tell() + prod(shape) * dtype.itemsize != size:
                raise ValueError("The NumPy file is truncated or has extra data after its tensor.")
    except (EOFError, TypeError, UnicodeError) as exc:
        raise ValueError("Cannot read this NumPy tensor file.") from exc
    return list(shape), dtype


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def import_array(source: Path, destination: Path, file_name: str) -> UploadedTensor:
    shape, dtype = inspect_array(source)
    data = np.load(source, mmap_mode="r", allow_pickle=False, max_header_size=MAX_HEADER_BYTES)
    # Preserve logical values and axis order while normalizing endian and storage layout.
    native = np.ascontiguousarray(data, dtype=dtype.newbyteorder("="))
    np.save(destination, native, allow_pickle=False)
    return UploadedTensor(
        id=destination.stem,
        file_name=file_name,
        sha256=checksum(destination),
        shape=shape,
        dtype=dtype.name,
        byte_count=destination.stat().st_size,
    )


def load_array(directory: Path, uploaded: UploadedTensor) -> np.ndarray:
    path = directory / f"{uploaded.id}.npy"
    if not path.is_file():
        raise ValueError(
            "The uploaded tensor file is missing. Upload it again or restore the inputs backup."
        )
    shape, dtype = inspect_array(path)
    if (
        shape != uploaded.shape
        or dtype.name != uploaded.dtype
        or path.stat().st_size != uploaded.byte_count
        or checksum(path) != uploaded.sha256
    ):
        raise ValueError(
            "The uploaded tensor file has changed. Upload it again to use the new values."
        )
    # A writable private copy prevents in-place model operations from changing the library.
    return np.array(np.load(path, mmap_mode="r", allow_pickle=False), copy=True, order="C")

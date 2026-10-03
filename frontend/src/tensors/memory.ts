import type { Tensor } from "../api/client";

const SIZES: Record<string, number> = {
  bool: 1,
  uint8: 1,
  int8: 1,
  float8_e4m3fn: 1,
  float8_e5m2: 1,
  int16: 2,
  uint16: 2,
  float16: 2,
  bfloat16: 2,
  int32: 4,
  uint32: 4,
  float32: 4,
  complex32: 4,
  int64: 8,
  uint64: 8,
  float64: 8,
  complex64: 8,
  complex128: 16,
};

/** Bytes per element of a dtype, or null when the dtype is not known. */
export function elementBytes(dtype: string): number | null {
  return SIZES[dtype.replace(/^torch\./, "")] ?? null;
}

/** Bytes the tensor's own elements take, not counting any shared storage. */
export function tensorBytes(tensor: Pick<Tensor, "dtype" | "numel">) {
  const size = elementBytes(tensor.dtype);
  return size === null ? null : size * tensor.numel;
}

/**
 * Bytes held by a set of tensors, counting views of one storage once (at the
 * largest of them), since they share their memory.
 */
export function storageBytes(
  tensors: Pick<Tensor, "dtype" | "numel" | "storage_id">[],
) {
  const storages = new Map<string, number>();
  for (const tensor of tensors) {
    const bytes = tensorBytes(tensor);
    if (bytes === null) continue;
    storages.set(
      tensor.storage_id,
      Math.max(storages.get(tensor.storage_id) ?? 0, bytes),
    );
  }
  return [...storages.values()].reduce((sum, bytes) => sum + bytes, 0);
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024,
    unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 100 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

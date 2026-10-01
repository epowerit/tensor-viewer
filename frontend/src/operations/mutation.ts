import type { Tensor } from "../api/client";

/**
 * Cells whose recorded value differs between two snapshots of one tensor.
 * Null when the snapshots cannot be compared cell by cell: different layouts,
 * shape-only runs, or values that are paged rather than held inline.
 */
export function changedCells(
  before: Tensor,
  after: Tensor,
  limit = 256,
): { indices: number[]; count: number } | null {
  if (
    before.dtype !== after.dtype ||
    before.shape.join() !== after.shape.join() ||
    !after.numel ||
    before.values.length !== before.numel ||
    after.values.length !== after.numel
  )
    return null;
  const indices: number[] = [];
  let count = 0;
  for (let i = 0; i < after.numel; i++) {
    if (before.values[i] === after.values[i]) continue;
    count++;
    if (indices.length < limit) indices.push(i);
  }
  return { indices, count };
}

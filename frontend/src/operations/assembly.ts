import { tensorSelection } from "./selection";
import type { Operation, Run, Tensor } from "../api/client";
import { product, ravel, unravel } from "../tensors/coordinates";

const joins = new Set(["cat", "concat", "concatenate", "stack"]);
const partitions = new Set(["split", "chunk", "unbind"]);
export type TensorAssembly = {
  joining: boolean;
  inserted: boolean;
  axis: number;
  whole: Tensor;
  parts: Tensor[];
  offsets: number[];
  ends: number[];
};
const equal = (a: number[], b: number[]) =>
  a.length === b.length && a.every((v, i) => v === b[i]);

/** Prove an ordered coordinate partition from actual operands and outputs. */
export function tensorAssembly(run: Run, op: Operation): TensorAssembly | null {
  if (
    (!joins.has(op.kind) && !partitions.has(op.kind)) ||
    op.status !== "ok" ||
    op.mutations?.length ||
    op.arguments.out != null
  )
    return null;
  const joining = joins.has(op.kind);
  const wholeIds = joining ? op.outputs : op.inputs;
  const partIds = joining ? op.inputs : op.outputs;
  if (wholeIds.length !== 1 || !partIds.length) return null;
  const whole = run.trace.tensors[wholeIds[0]],
    parts = partIds.map((id) => run.trace.tensors[id]);
  if (
    !whole ||
    !whole.numel ||
    !whole.shape.length ||
    parts.some((t) => !t || t.dtype !== whole.dtype)
  )
    return null;
  if (
    [whole, ...parts].some(
      (t) =>
        t.shape.some((n) => !Number.isSafeInteger(n) || n < 0) ||
        !Number.isSafeInteger(t.numel) ||
        product(t.shape) !== t.numel,
    )
  )
    return null;
  // Old traces named positional arguments arg1/arg2. Keyword/default dim stays explicit.
  const legacy = ["split", "chunk"].includes(op.kind) ? "arg2" : "arg1";
  const raw = op.arguments.dim ?? op.arguments[legacy] ?? 0;
  const rank = whole.shape.length;
  if (
    typeof raw !== "number" ||
    !Number.isInteger(raw) ||
    raw < -rank ||
    raw >= rank
  )
    return null;
  const axis = (raw + rank) % rank;
  const inserted = ["stack", "unbind"].includes(op.kind);
  if (inserted) {
    const expected = whole.shape.filter((_, i) => i !== axis);
    if (
      parts.length !== whole.shape[axis] ||
      parts.some((p) => !equal(p.shape, expected))
    )
      return null;
  } else if (
    parts.some(
      (p) =>
        p.shape.length !== rank ||
        p.shape.some((n, i) => i !== axis && n !== whole.shape[i]),
    ) ||
    parts.reduce((n, p) => n + p.shape[axis], 0) !== whole.shape[axis]
  )
    return null;
  let offset = 0;
  const offsets: number[] = [],
    ends: number[] = [];
  for (const part of parts) {
    offsets.push(offset);
    offset += inserted ? 1 : part.shape[axis];
    ends.push(offset);
  }
  return { joining, inserted, axis, whole, parts, offsets, ends };
}

/** O(log(parts) + rank); never build a map proportional to the number of cells. */
export function locatePart(p: TensorAssembly, index: number) {
  const coordinates = unravel(index, p.whole.shape),
    coordinate = coordinates[p.axis];
  let low = 0,
    high = p.ends.length - 1;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (p.ends[mid] > coordinate) high = mid;
    else low = mid + 1;
  }
  const local = [...coordinates];
  if (p.inserted) local.splice(p.axis, 1);
  else local[p.axis] -= p.offsets[low];
  return {
    part: low,
    index: ravel(local, p.parts[low].shape),
    coordinates: local,
  };
}
export function locateWhole(p: TensorAssembly, part: number, index: number) {
  const coordinates = unravel(index, p.parts[part].shape);
  if (p.inserted) coordinates.splice(p.axis, 0, part);
  else coordinates[p.axis] += p.offsets[part];
  return { index: ravel(coordinates, p.whole.shape), coordinates };
}
export function visibleParts(count: number, selected: number): number[] {
  if (count <= 6) return Array.from({ length: count }, (_, i) => i);
  return [...new Set([0, 1, selected - 1, selected, selected + 1, count - 1])]
    .filter((i) => i >= 0 && i < count)
    .sort((a, b) => a - b);
}

/** Open the chosen split result, or map a joined result cell back to its part. */
export function assemblySelection(
  p: TensorAssembly,
  tensorId?: string,
  cell?: number,
) {
  if (p.joining) {
    const { index } = tensorSelection([p.whole], tensorId, cell);
    return locatePart(p, index);
  }
  const requested = p.parts.find((part) => part.id === tensorId);
  const fallback = p.parts.find((part) => part.numel > 0) ?? p.parts[0];
  const { choice, index } = tensorSelection(
    p.parts,
    (requested ?? fallback).id,
    cell,
  );
  return { part: choice, index };
}

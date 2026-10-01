import type { Operation, Run, Tensor } from "../api/client";
import { product, ravel, unravel } from "../tensors/coordinates";

const floats = ["float16", "bfloat16", "float32", "float64"];
const integers = ["uint8", "int8", "int16", "int32", "int64"];
const integral = (dtype: string) =>
  integers.includes(dtype) || dtype === "bool";

export type Reduction = {
  operationId: string;
  kind: "mean" | "sum";
  input: Tensor;
  output: Tensor;
  axes: number[];
  keepdim: boolean;
  size: number;
  cast: boolean;
};

/** Independently validate imported and older runs before advertising a relationship. */
export function tensorReduction(run: Run, op: Operation): Reduction | null {
  if (
    (op.kind !== "mean" && op.kind !== "sum") ||
    op.status !== "ok" ||
    op.mutations?.length ||
    op.inputs.length !== 1 ||
    op.outputs.length !== 1 ||
    op.arguments.out != null
  )
    return null;
  const input = run.trace.tensors[op.inputs[0]],
    output = run.trace.tensors[op.outputs[0]];
  if (
    [input, output].some(
      (t) =>
        !t ||
        !Number.isSafeInteger(t.numel) ||
        t.numel < 1 ||
        t.shape.some((n) => !Number.isSafeInteger(n) || n < 1) ||
        product(t.shape) !== t.numel,
    )
  )
    return null;
  const requested = op.arguments.dtype;
  const dtype =
    requested == null
      ? op.kind === "sum" && integral(input.dtype)
        ? "int64"
        : input.dtype
      : typeof requested === "string" && requested.startsWith("torch.")
        ? requested.slice(6)
        : null;
  if (
    ![...floats, ...integers, "bool"].includes(input.dtype) ||
    !dtype ||
    !(op.kind === "mean" ? floats : [...floats, ...integers]).includes(dtype) ||
    output.dtype !== dtype
  )
    return null;
  const keepdim = op.arguments.keepdim ?? false;
  // An explicit null is not the default boolean accepted by PyTorch.
  if (typeof keepdim !== "boolean" || op.arguments.keepdim === null)
    return null;
  const rank = input.shape.length,
    dim = op.arguments.dim;
  let axes: number[];
  if (dim == null || (Array.isArray(dim) && !dim.length))
    axes = input.shape.map((_, i) => i);
  else {
    const dims = Array.isArray(dim) ? dim : [dim],
      virtualRank = Math.max(rank, 1);
    if (
      !dims.every(
        (d): d is number =>
          typeof d === "number" &&
          Number.isInteger(d) &&
          d >= -virtualRank &&
          d < virtualRank,
      )
    )
      return null;
    const normalized = dims.map((d) => (d + virtualRank) % virtualRank);
    if (new Set(normalized).size !== normalized.length) return null;
    axes = rank ? normalized.sort((a, b) => a - b) : [];
  }
  const shape = keepdim
    ? input.shape.map((n, i) => (axes.includes(i) ? 1 : n))
    : input.shape.filter((_, i) => !axes.includes(i));
  if (
    shape.length !== output.shape.length ||
    shape.some((n, i) => n !== output.shape[i])
  )
    return null;
  return {
    operationId: op.id,
    kind: op.kind,
    input,
    output,
    axes,
    keepdim,
    size: product(axes.map((i) => input.shape[i])),
    cast:
      input.dtype !== output.dtype &&
      !(integral(input.dtype) && output.dtype === "int64"),
  };
}

/** Logical coordinates, including non-contiguous snapshots. Work scales with rank. */
export function reductionSelection(p: Reduction, output: number, term: number) {
  const outputCoordinates = unravel(output, p.output.shape);
  const reduced = unravel(
    term,
    p.axes.map((i) => p.input.shape[i]),
  );
  let kept = 0;
  const inputCoordinates = p.input.shape.map((_, i) => {
    const axis = p.axes.indexOf(i);
    return axis >= 0
      ? reduced[axis]
      : outputCoordinates[p.keepdim ? i : kept++];
  });
  return {
    output,
    term,
    outputCoordinates,
    inputCoordinates,
    input: ravel(inputCoordinates, p.input.shape),
  };
}

export function reductionInputSelection(p: Reduction, input: number) {
  const coordinates = unravel(input, p.input.shape);
  const output = p.keepdim
    ? coordinates.map((n, i) => (p.axes.includes(i) ? 0 : n))
    : coordinates.filter((_, i) => !p.axes.includes(i));
  return {
    output: ravel(output, p.output.shape),
    term: ravel(
      p.axes.map((i) => coordinates[i]),
      p.axes.map((i) => p.input.shape[i]),
    ),
  };
}

/** Eight visible contributors; optional full small group never exceeds 256 indices. */
export function reductionWindow(
  p: Reduction,
  output: number,
  term: number,
  full = false,
) {
  const complete = full && p.size <= 256;
  const start = complete ? 0 : Math.floor(term / 8) * 8;
  return Array.from(
    { length: complete ? p.size : Math.min(8, p.size - start) },
    (_, i) => reductionSelection(p, output, start + i),
  );
}

const finite = (n: unknown): n is number =>
  typeof n === "number" && Number.isFinite(n);
// Compensated sums are explanatory, not an emulation of PyTorch's accumulation order.
function sum(values: number[]) {
  let total = 0,
    correction = 0;
  for (const value of values) {
    const next = total + value;
    correction +=
      Math.abs(total) >= Math.abs(value)
        ? total - next + value
        : value - next + total;
    total = next;
  }
  const result = total + correction;
  return Number.isFinite(result) ? result : undefined;
}

/** Never substitute a window sum for a whole reduction. Preserve large integers exactly. */
export function reductionReference(
  p: Reduction,
  values: unknown[],
): { sum: number | string | undefined; result: number | string } | null {
  if (p.cast || p.size > 256 || values.length !== p.size) return null;
  if (integral(p.input.dtype)) {
    let total = 0n;
    for (const value of values) {
      if (typeof value === "number" && Number.isSafeInteger(value))
        total += BigInt(value);
      else if (typeof value === "string" && /^-?\d{1,20}$/.test(value))
        total += BigInt(value);
      else return null;
    }
    const result =
      total > BigInt(Number.MAX_SAFE_INTEGER) ||
      total < BigInt(Number.MIN_SAFE_INTEGER)
        ? String(total)
        : Number(total);
    return { sum: result, result };
  }
  if (!values.every(finite)) return null;
  const total = sum(values);
  const result =
    p.kind === "mean" ? sum(values.map((value) => value / p.size)) : total;
  return result === undefined ? null : { sum: total, result };
}

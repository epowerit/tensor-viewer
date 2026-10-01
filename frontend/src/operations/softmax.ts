import type { Operation, Run, SoftmaxStatistics, Tensor } from "../api/client";
import { product, unravel } from "../tensors/coordinates";

export type Softmax = {
  operationId: string;
  input: Tensor;
  output: Tensor;
  axis: number;
  size: number;
  groups: number;
  stride: number;
};
export type SoftmaxSummary = Pick<
  SoftmaxStatistics,
  "status" | "maximum" | "denominator" | "masked_count"
>;

export function tensorSoftmax(run: Run, op: Operation): Softmax | null {
  if (
    op.kind !== "softmax" ||
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
  const dim = op.arguments.dim,
    rank = Math.max(1, input.shape.length),
    dtype = op.arguments.dtype;
  if (
    typeof dim !== "number" ||
    !Number.isInteger(dim) ||
    dim < -rank ||
    dim >= rank ||
    input.shape.length !== output.shape.length ||
    input.shape.some((n, i) => n !== output.shape[i]) ||
    !["float16", "bfloat16", "float32", "float64"].includes(input.dtype) ||
    input.dtype !== output.dtype ||
    (dtype != null && dtype !== `torch.${input.dtype}`)
  )
    return null;
  const axis = (dim + rank) % rank,
    size = input.shape[axis] ?? 1;
  return {
    operationId: op.id,
    input,
    output,
    axis,
    size,
    groups: input.numel / size,
    stride: product(input.shape.slice(axis + 1)),
  };
}

export function softmaxIndex(p: Softmax, group: number, member: number) {
  return (
    Math.floor(group / p.stride) * p.size * p.stride +
    (group % p.stride) +
    member * p.stride
  );
}
export function softmaxSelection(p: Softmax, selected: number) {
  return {
    group:
      Math.floor(selected / (p.size * p.stride)) * p.stride +
      (selected % p.stride),
    member: Math.floor(selected / p.stride) % p.size,
    coordinates: unravel(selected, p.input.shape),
  };
}
/** Logical groups can cross rows, slices, or batches. No tensor-sized map. */
export function softmaxWindow(p: Softmax, selected: number, full = false) {
  const { group, member } = softmaxSelection(p, selected);
  const start = full && p.size <= 256 ? 0 : Math.floor(member / 8) * 8;
  const count = full && p.size <= 256 ? p.size : Math.min(8, p.size - start);
  return Array.from({ length: count }, (_, i) => ({
    member: start + i,
    index: softmaxIndex(p, group, start + i),
  }));
}
const score = (value: unknown): number | undefined =>
  typeof value === "number"
    ? value
    : value === "-inf"
      ? -Infinity
      : value === "inf"
        ? Infinity
        : value === "nan"
          ? NaN
          : undefined;

export function softmaxSummary(
  values: unknown[],
  size: number,
): SoftmaxSummary | null {
  if (!size || values.length !== size) return null;
  const numbers = values.map(score);
  if (numbers.some((x) => x === undefined)) return null;
  const xs = numbers as number[];
  if (xs.some((x) => Number.isNaN(x) || x === Infinity))
    return { status: "non_finite" };
  const masked = xs.filter((x) => x === -Infinity).length;
  if (masked === size) return { status: "all_masked", masked_count: masked };
  const maximum = Math.max(...xs);
  let denominator = 0,
    correction = 0;
  for (const x of xs) {
    const term = Math.exp(x - maximum) - correction,
      next = denominator + term;
    correction = next - denominator - term;
    denominator = next;
  }
  return { status: "ok", maximum, denominator, masked_count: masked };
}

export function softmaxCalculation(
  summary: SoftmaxSummary | null | undefined,
  value: unknown,
) {
  const x = score(value);
  if (
    summary?.status !== "ok" ||
    summary.maximum == null ||
    summary.denominator == null ||
    x === undefined ||
    Number.isNaN(x) ||
    x === Infinity ||
    x > summary.maximum
  )
    return null;
  const shifted = x - summary.maximum,
    exponential = Math.exp(shifted);
  return { shifted, exponential, weight: exponential / summary.denominator };
}

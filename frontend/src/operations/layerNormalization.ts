import type { Operation, Run, Tensor } from "../api/client";
import { product, unravel } from "../tensors/coordinates";

export type LayerNormalization = {
  operationId: string;
  input: Tensor;
  output: Tensor;
  weight?: Tensor;
  bias?: Tensor;
  shape: number[];
  size: number;
  groups: number;
  eps: number;
};
const same = (a: number[], b: number[]) =>
  a.length === b.length && a.every((n, i) => n === b[i]);
const finite = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);

export function layerNormalization(
  run: Run,
  op: Operation,
): LayerNormalization | null {
  if (
    op.kind !== "layer_norm" ||
    op.status !== "ok" ||
    op.mutations?.length ||
    !op.inputs.length ||
    op.inputs.length > 3 ||
    op.outputs.length !== 1
  )
    return null;
  const inputs = op.inputs.map((id) => run.trace.tensors[id]),
    input = inputs[0],
    output = run.trace.tensors[op.outputs[0]];
  if (
    [...inputs, output].some(
      (t) =>
        !t ||
        !Number.isSafeInteger(t.numel) ||
        t.numel < 1 ||
        t.shape.some((n) => !Number.isSafeInteger(n) || n < 1) ||
        product(t.shape) !== t.numel,
    )
  )
    return null;
  const shape = op.arguments.normalized_shape,
    eps = Object.hasOwn(op.arguments, "eps") ? op.arguments.eps : 1e-5;
  if (
    !Array.isArray(shape) ||
    !shape.length ||
    shape.length > input.shape.length ||
    !shape.every(
      (n) => typeof n === "number" && Number.isSafeInteger(n) && n > 0,
    ) ||
    !same(shape, input.shape.slice(-shape.length)) ||
    !same(input.shape, output.shape) ||
    !["float16", "bfloat16", "float32", "float64"].includes(input.dtype) ||
    [...inputs, output].some((t) => t.dtype !== input.dtype) ||
    !finite(eps) ||
    eps < 0
  )
    return null;
  const roles: (boolean | undefined)[] = [];
  for (const name of ["weight", "bias"]) {
    if (!Object.hasOwn(op.arguments, name)) roles.push(undefined);
    else if (op.arguments[name] === "tensor") roles.push(true);
    else if (op.arguments[name] === null) roles.push(false);
    else return null;
  }
  const missing = roles.filter((r) => r === undefined).length;
  const remaining = inputs.length - 1 - roles.filter((r) => r === true).length;
  if (
    remaining < 0 ||
    remaining > missing ||
    (missing === 2 && remaining === 1)
  )
    return null;
  const [hasWeight, hasBias] = roles.map((r) => r ?? remaining > 0);
  if (inputs.slice(1).some((t) => !same(t.shape, shape))) return null;
  const size = product(shape);
  return {
    operationId: op.id,
    input,
    output,
    shape,
    size,
    groups: input.numel / size,
    eps,
    weight: hasWeight ? inputs[1] : undefined,
    bias: hasBias ? inputs[hasWeight ? 2 : 1] : undefined,
  };
}

export function layerNormSelection(p: LayerNormalization, selected: number) {
  const group = Math.floor(selected / p.size),
    feature = selected % p.size;
  return {
    group,
    feature,
    start: group * p.size,
    coordinates: unravel(selected, p.input.shape),
    prefix: unravel(selected, p.input.shape).slice(0, -p.shape.length),
    parameterCoordinates: unravel(feature, p.shape),
  };
}

/** Group navigation is independent of tensor size and physical storage layout. */
export function layerNormWindow(
  p: LayerNormalization,
  selected: number,
  full = false,
) {
  const { start, feature } = layerNormSelection(p, selected);
  const offset = full && p.size <= 256 ? 0 : Math.floor(feature / 8) * 8;
  const count = full && p.size <= 256 ? p.size : Math.min(8, p.size - offset);
  return Array.from({ length: count }, (_, i) => start + offset + i);
}

/** Two-pass population statistics. Missing, partial, or non-finite groups have no result. */
export function layerNormStatistics(
  values: unknown[],
  size: number,
  eps: number,
) {
  if (
    !size ||
    values.length !== size ||
    !values.every(finite) ||
    !finite(eps) ||
    eps < 0
  )
    return null;
  // Shift by one input to preserve small differences and keep constant groups exact.
  const origin = values[0];
  const mean =
    origin + values.reduce((total, x) => total + (x - origin) / size, 0);
  const variance = values.reduce(
    (total, x) => total + (x - mean) ** 2 / size,
    0,
  );
  const denominator = Math.sqrt(variance + eps);
  return [mean, variance, denominator].every(Number.isFinite)
    ? { mean, variance, denominator }
    : null;
}

export function layerNormCalculation(
  statistics: ReturnType<typeof layerNormStatistics>,
  value: unknown,
  scale: unknown,
  bias: unknown,
) {
  if (!statistics || !finite(value) || statistics.denominator === 0)
    return null;
  const centered = value - statistics.mean;
  const normalized = centered / statistics.denominator;
  if (![centered, normalized].every(Number.isFinite)) return null;
  const output =
    finite(scale) && finite(bias) ? normalized * scale + bias : undefined;
  return { centered, normalized, output: finite(output) ? output : undefined };
}

import type { Operation, Run, Tensor } from "../api/client";
import { product, ravel, unravel } from "../tensors/coordinates";
import { nearestWindowSample } from "../tensors/windowGeometry";
import { contributionSum } from "./linear";

const kinds = new Set([
  "max_pool2d",
  "max_pool2d_with_indices",
  "avg_pool2d",
  "adaptive_avg_pool2d",
]);
const integer = (v: unknown, min: number): v is number =>
  typeof v === "number" && Number.isSafeInteger(v) && v >= min;
function pair(v: unknown, minimum: number): number[] | null {
  const a =
    typeof v === "number"
      ? [v, v]
      : Array.isArray(v) && v.length === 1
        ? [v[0], v[0]]
        : v;
  return Array.isArray(a) &&
    a.length === 2 &&
    a.every((n) => integer(n, minimum))
    ? (a as number[])
    : null;
}
export type Pooling = {
  input: Tensor;
  output: Tensor;
  indices?: Tensor;
  mode: "max" | "average" | "adaptive";
  kernel: number[];
  stride: number[];
  padding: number[];
  dilation: number[];
  ceil: boolean;
  includePadding: boolean;
  divisor: number | null;
};
export function tensorPooling(run: Run, op: Operation): Pooling | null {
  const indexed = op.kind === "max_pool2d_with_indices";
  if (
    !kinds.has(op.kind) ||
    op.status !== "ok" ||
    op.mutations?.length ||
    op.inputs.length !== 1 ||
    op.outputs.length !== (indexed ? 2 : 1)
  )
    return null;
  const input = run.trace.tensors[op.inputs[0]],
    output = run.trace.tensors[op.outputs[0]],
    indices = indexed ? run.trace.tensors[op.outputs[1]] : undefined;
  if (
    !input ||
    !output ||
    ![3, 4].includes(input.shape.length) ||
    output.shape.length !== input.shape.length ||
    !["float16", "bfloat16", "float32", "float64"].includes(input.dtype) ||
    output.dtype !== input.dtype ||
    input.shape.slice(0, -2).some((n, i) => output.shape[i] !== n) ||
    (indexed &&
      (!indices ||
        indices.dtype !== "int64" ||
        indices.shape.length !== output.shape.length ||
        indices.shape.some((n, i) => n !== output.shape[i]))) ||
    [input, output, ...(indices ? [indices] : [])].some(
      (t) =>
        !integer(t.numel, 1) ||
        t.shape.some((n) => !integer(n, 1)) ||
        product(t.shape) !== t.numel,
    )
  )
    return null;
  const arg = (name: string, position: number, fallback: unknown) =>
    Object.hasOwn(op.arguments, name)
      ? op.arguments[name]
      : Object.hasOwn(op.arguments, `arg${position}`)
        ? op.arguments[`arg${position}`]
        : fallback;
  const mode =
    op.kind === "adaptive_avg_pool2d"
      ? "adaptive"
      : op.kind.startsWith("max_")
        ? "max"
        : "average";
  if (mode === "adaptive") {
    const size = arg("output_size", 1, undefined),
      values = typeof size === "number" ? [size, size] : size;
    if (!Array.isArray(values) || values.length !== 2) return null;
    const expected = values.map((n, i) =>
      n === null ? input.shape.at(-2 + i)! : n,
    );
    if (
      expected.some((n, i) => !integer(n, 1) || n !== output.shape.at(-2 + i))
    )
      return null;
    return {
      input,
      output,
      mode,
      kernel: [],
      stride: [],
      padding: [0, 0],
      dilation: [1, 1],
      ceil: false,
      includePadding: false,
      divisor: null,
    };
  }
  const kernel = pair(arg("kernel_size", 1, undefined), 1);
  const rawStride = arg("stride", 2, null);
  const stride =
    rawStride === null || (Array.isArray(rawStride) && !rawStride.length)
      ? kernel
      : pair(rawStride, 1);
  const padding = pair(arg("padding", 3, 0), 0);
  const dilation = mode === "max" ? pair(arg("dilation", 4, 1), 1) : [1, 1];
  const ceil = arg("ceil_mode", mode === "max" ? 5 : 4, false),
    includePadding = mode === "max" ? false : arg("count_include_pad", 5, true),
    divisor = mode === "max" ? null : arg("divisor_override", 6, null);
  if (
    !kernel ||
    !stride ||
    !padding ||
    !dilation ||
    typeof ceil !== "boolean" ||
    typeof includePadding !== "boolean" ||
    !integer(product(kernel), 1) ||
    padding.some((n, i) => n > Math.floor(kernel[i] / 2)) ||
    (divisor !== null &&
      (typeof divisor !== "number" ||
        !Number.isSafeInteger(divisor) ||
        divisor === 0))
  )
    return null;
  const effective = kernel.map((k, i) => dilation[i] * (k - 1) + 1);
  if (effective.some((n) => !integer(n, 1))) return null;
  const expected = input.shape.slice(-2).map((n, i) => {
    const padded = n + 2 * padding[i] + (ceil ? stride[i] - 1 : 0);
    if (!integer(padded, 1)) return NaN;
    let size = Math.floor((padded - effective[i]) / stride[i]) + 1;
    if (ceil && (size - 1) * stride[i] >= n + padding[i]) size--;
    return size;
  });
  if (expected.some((n, i) => n !== output.shape.at(-2 + i))) return null;
  return {
    input,
    output,
    indices,
    mode,
    kernel,
    stride,
    padding,
    dilation,
    ceil,
    includePadding,
    divisor: divisor as number | null,
  };
}
// Keep products exact even when each dimension is safe but their product is not.
const floorRatio = (a: number, b: number, c: number) =>
  Number((BigInt(a) * BigInt(b)) / BigInt(c));
const ceilRatio = (a: number, b: number, c: number) =>
  Number((BigInt(a) * BigInt(b) + BigInt(c) - 1n) / BigInt(c));
export function poolingRegion(p: Pooling, output: number) {
  const coordinates = unravel(output, p.output.shape),
    position = coordinates.slice(-2),
    spatial = p.input.shape.slice(-2);
  const start = position.map((o, i) =>
    p.mode === "adaptive"
      ? floorRatio(o, spatial[i], p.output.shape.at(-2 + i)!)
      : o * p.stride[i] - p.padding[i],
  );
  const shape = start.map((s, i) =>
    p.mode === "adaptive"
      ? ceilRatio(position[i] + 1, spatial[i], p.output.shape.at(-2 + i)!) - s
      : p.mode === "average"
        ? Math.min(p.kernel[i], spatial[i] + p.padding[i] - s)
        : p.kernel[i],
  );
  const validCounts = start.map((s, i) =>
    Math.max(
      0,
      Math.min(shape[i] - 1, Math.floor((spatial[i] - 1 - s) / p.dilation[i])) -
        Math.max(0, Math.ceil(-s / p.dilation[i])) +
        1,
    ),
  );
  const terms = product(shape),
    real = product(validCounts);
  return {
    output,
    coordinates,
    start,
    shape,
    terms,
    real,
    padded: terms - real,
    divisor:
      p.mode === "max"
        ? null
        : (p.divisor ?? (p.includePadding ? terms : real)),
  };
}
export type PoolRegion = ReturnType<typeof poolingRegion>;
export function poolingSample(p: Pooling, region: PoolRegion, term: number) {
  const offsets = unravel(term, region.shape);
  const spatial = offsets.map((n, i) => region.start[i] + n * p.dilation[i]);
  const coordinates = [...region.coordinates.slice(0, -2), ...spatial];
  const padding = spatial.some(
    (n, i) => n < 0 || n >= p.input.shape.at(-2 + i)!,
  );
  return {
    term,
    coordinates,
    padding,
    input: padding ? null : ravel(coordinates, p.input.shape),
  };
}
export function poolingSamples(
  p: Pooling,
  region: PoolRegion,
  term: number,
  full = false,
) {
  const start = full && region.terms <= 256 ? 0 : Math.floor(term / 8) * 8;
  const count =
    full && region.terms <= 256
      ? region.terms
      : Math.min(8, region.terms - start);
  return Array.from({ length: count }, (_, i) =>
    poolingSample(p, region, start + i),
  );
}
export function poolingInputSelection(
  p: Pooling,
  input: number,
  output: number,
) {
  const source = unravel(input, p.input.shape),
    target = unravel(output, p.output.shape);
  const position: number[] = [];
  for (let i = 0; i < 2; i++) {
    const x = source.at(-2 + i)!,
      previous = target.at(-2 + i)!;
    if (p.mode === "adaptive") {
      const n = p.input.shape.at(-2 + i)!,
        m = p.output.shape.at(-2 + i)!;
      const lo = floorRatio(x, m, n),
        hi = ceilRatio(x + 1, m, n) - 1;
      position.push(Math.max(lo, Math.min(hi, previous)));
    } else {
      const sample = nearestWindowSample(
        x,
        p.padding[i],
        p.stride[i],
        p.dilation[i],
        p.kernel[i],
        p.output.shape.at(-2 + i)!,
        previous,
      );
      if (!sample) return null;
      position.push(sample.output);
    }
  }
  const next = ravel([...source.slice(0, -2), ...position], p.output.shape),
    region = poolingRegion(p, next);
  const local = source
    .slice(-2)
    .map((n, i) => (n - region.start[i]) / p.dilation[i]);
  return { output: next, term: ravel(local, region.shape) };
}
export function poolingWinner(
  p: Pooling,
  region: PoolRegion,
  index: number | string | undefined,
) {
  if (
    !p.indices ||
    !integer(index, 0) ||
    index >= product(p.input.shape.slice(-2))
  )
    return null;
  const spatial = unravel(index, p.input.shape.slice(-2));
  const local = spatial.map((n, i) => (n - region.start[i]) / p.dilation[i]);
  if (local.some((n, i) => !integer(n, 0) || n >= region.shape[i])) return null;
  return poolingSample(p, region, ravel(local, region.shape));
}
export function poolingCalculation(
  p: Pooling,
  region: PoolRegion,
  samples: ReturnType<typeof poolingSamples>,
  valueAt: (index: number) => number | string | undefined,
  recorded: number | string | undefined,
) {
  const complete = samples.length === region.terms;
  const real = samples.filter((s) => s.input !== null);
  const values = real.map((s) => valueAt(s.input!));
  const finite = values.every(
    (v) => typeof v === "number" && Number.isFinite(v),
  );
  let aggregate: number | undefined;
  if (finite)
    aggregate =
      p.mode === "max"
        ? values.length
          ? Math.max(...(values as number[]))
          : undefined
        : values.length
          ? contributionSum(values as number[])
          : 0;
  const result =
    complete && aggregate !== undefined
      ? p.mode === "max"
        ? aggregate
        : aggregate / region.divisor!
      : undefined;
  return {
    complete,
    aggregate,
    result:
      result !== undefined && Number.isFinite(result) ? result : undefined,
    matches:
      p.mode === "max" &&
      complete &&
      aggregate !== undefined &&
      aggregate === recorded
        ? real
            .filter((s) => valueAt(s.input!) === recorded)
            .map((s) => s.input!)
        : [],
  };
}

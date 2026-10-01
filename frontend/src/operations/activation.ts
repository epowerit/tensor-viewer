import type { Operation, Run, Tensor } from "../api/client";
import { product } from "../tensors/coordinates";

export const activationNames = {
  relu: "ReLU",
  gelu: "GELU",
  sigmoid: "Sigmoid",
  tanh: "Tanh",
};
export type ActivationKind = keyof typeof activationNames;
export type Activation = {
  kind: ActivationKind;
  approximate: "none" | "tanh";
  input: Tensor;
  output: Tensor;
};
export const finiteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

/** Same logical coordinate is verified from the recorded operands, never inferred from labels. */
export function tensorActivation(run: Run, op: Operation): Activation | null {
  if (
    !Object.hasOwn(activationNames, op.kind) ||
    op.status !== "ok" ||
    op.mutations?.length ||
    op.inputs.length !== 1 ||
    op.outputs.length !== 1 ||
    op.arguments.inplace ||
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
  const kind = op.kind as ActivationKind;
  const dtypes = [
    "float16",
    "bfloat16",
    "float32",
    "float64",
    ...(kind === "relu" ? ["uint8", "int8", "int16", "int32", "int64"] : []),
  ];
  if (
    input.dtype !== output.dtype ||
    !dtypes.includes(input.dtype) ||
    input.shape.length !== output.shape.length ||
    input.shape.some((n, i) => n !== output.shape[i])
  )
    return null;
  const approximate =
    kind === "gelu" && Object.hasOwn(op.arguments, "approximate")
      ? op.arguments.approximate
      : "none";
  if (approximate !== "none" && approximate !== "tanh") return null;
  return { kind, approximate, input, output };
}

export function activationWindow(total: number, selected: number) {
  const start = Math.floor(selected / 8) * 8;
  return Array.from(
    { length: Math.min(8, total - start) },
    (_, i) => start + i,
  );
}

export function activationFormula(p: Pick<Activation, "kind" | "approximate">) {
  return {
    relu: "max(0, x)",
    gelu:
      p.approximate === "tanh"
        ? "½x · (1 + tanh(√(2/π) · (x + 0.044715x³)))"
        : "x · Φ(x)",
    sigmoid: "1 / (1 + exp(−x))",
    tanh: "tanh(x)",
  }[p.kind];
}

/** Explanatory reference only. Captured PyTorch outputs are never replaced. */
export function activationValue(
  p: Pick<Activation, "kind" | "approximate">,
  x: unknown,
): number | undefined {
  if (!finiteNumber(x)) return undefined;
  if (p.kind === "relu") return Math.max(0, x);
  if (p.kind === "tanh") return Math.tanh(x);
  if (p.kind === "sigmoid") {
    const exp = Math.exp(-Math.abs(x));
    return x >= 0 ? 1 / (1 + exp) : exp / (1 + exp);
  }
  if (x === 0) return 0;
  // At these tails the correction is below browser display precision. Avoid x³ overflow.
  if (Math.abs(x) >= 10) return x > 0 ? x : 0;
  if (p.approximate === "tanh")
    return (
      x *
      (0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))))
    );
  // Five-term erf approximation (absolute erf error < 1.5e-7).
  // Form the small tail directly, avoiding 1 − erf cancellation for negative inputs.
  const z = Math.abs(x) / Math.SQRT2,
    t = 1 / (1 + 0.3275911 * z);
  const tail =
    0.5 *
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
      t +
      0.254829592) *
    t *
    Math.exp(-z * z);
  return x * (x < 0 ? tail : 1 - tail);
}

export type ActivationPair = { index: number; input: unknown; output: unknown };
/** Fixed work regardless of tensor size; extreme domains scale before subtracting. */
export function activationPlot(
  p: Pick<Activation, "kind" | "approximate">,
  pairs: ActivationPair[],
  fit: boolean,
) {
  const finite = pairs.filter(
    (pair): pair is { index: number; input: number; output: number } =>
      finiteNumber(pair.input) && finiteNumber(pair.output),
  );
  const xmin = fit ? Math.min(-4, ...finite.map((pair) => pair.input)) : -4;
  const xmax = fit ? Math.max(4, ...finite.map((pair) => pair.input)) : 4;
  // Keep zero and the nonlinear neighborhood even when a large value expands
  // the range. Otherwise a coarse fitted ReLU segment could miss its corner.
  const domain = new Set([
    ...Array.from(
      { length: 201 },
      (_, i) => xmin * (1 - i / 200) + xmax * (i / 200),
    ),
    ...Array.from({ length: 41 }, (_, i) => (i - 20) / 2),
  ]);
  const samples = [...domain]
    .filter((x) => x >= xmin && x <= xmax)
    .sort((a, b) => a - b)
    .map((x) => ({ x, y: activationValue(p, x)! }));
  const visible = finite.filter(
    (pair) => pair.input >= xmin && pair.input <= xmax,
  );
  const ymin = Math.min(
    p.kind === "tanh" ? -1.25 : -0.25,
    ...samples.map((point) => point.y),
    ...visible.map((pair) => pair.output),
  );
  const ymax = Math.max(
    1.25,
    ...samples.map((point) => point.y),
    ...visible.map((pair) => pair.output),
  );
  const ratio = (value: number, min: number, max: number) => {
    const scale = Math.max(Math.abs(min), Math.abs(max), 1);
    return (value / scale - min / scale) / (max / scale - min / scale);
  };
  const project = (x: number, y: number) => ({
    x: 58 + 548 * ratio(x, xmin, xmax),
    y: 250 - 214 * ratio(y, ymin, ymax),
  });
  return {
    xmin,
    xmax,
    ymin,
    ymax,
    project,
    samples: samples.map((point) => project(point.x, point.y)),
    points: visible.map((pair) => ({
      ...pair,
      ...project(pair.input, pair.output),
    })),
    omitted: finite.length - visible.length,
  };
}

import type { Run, Tensor } from "../api/client";
import { stepFlops } from "./cost";
import { liveMemory } from "./liveMemory";

type Trace = Run["trace"];

/** A symbol as symbolic shapes write it: `B`, `T`, `T'`, `N3`. */
const SYMBOL = /[A-Z][0-9]*'*/g;

/**
 * A symbolic size (`B·T`, `3·T`, `T/2`, `T+2`, `2·T²`, or a plain number) at
 * the given symbol values; null when it names a symbol without a value or is
 * not a size symbolic shapes write.
 */
export function evaluateSize(
  term: string,
  values: Record<string, number>,
): number | null {
  let missing = false;
  const text = term
    .replace(/·/g, "*")
    .replace(/−/g, "-")
    .replace(/²/g, "^2")
    .replace(SYMBOL, (symbol) => {
      if (!(symbol in values)) missing = true;
      return `(${values[symbol] ?? 0})`;
    });
  if (missing || !/^[\d\s()+\-*/^.]+$/.test(text)) return null;
  // A small parser: sums of products of powers of numbers and groups.
  let at = 0;
  const peek = () => text[at];
  const skip = () => {
    while (text[at] === " ") at++;
  };
  const atom = (): number => {
    skip();
    if (peek() === "(") {
      at++;
      const value = sum();
      skip();
      at++; // ")"
      return value;
    }
    const match = /^\d+(\.\d+)?/.exec(text.slice(at));
    if (!match) throw new Error("number");
    at += match[0].length;
    return Number(match[0]);
  };
  const power = (): number => {
    const base = atom();
    skip();
    if (peek() === "^") {
      at++;
      return base ** atom();
    }
    return base;
  };
  const product = (): number => {
    let value = power();
    for (skip(); peek() === "*" || peek() === "/"; skip()) {
      const op = text[at++];
      value = op === "*" ? value * power() : value / power();
    }
    return value;
  };
  const sum = (): number => {
    let value = product();
    for (skip(); peek() === "+" || peek() === "-"; skip()) {
      const op = text[at++];
      value = op === "+" ? value + product() : value - product();
    }
    return value;
  };
  try {
    const value = sum();
    skip();
    return at === text.length && Number.isFinite(value) ? value : null;
  } catch {
    return null;
  }
}

/** Each symbol's size in the recorded run, read off the input's shape. */
export function recordedSymbols(
  input: Tensor | undefined,
  labels: (string | null)[] | null,
): Record<string, number> {
  const sizes: Record<string, number> = {};
  if (!input || !labels) return sizes;
  labels.forEach((label, axis) => {
    if (label && /^[A-Z][0-9]*'*$/.test(label))
      sizes[label] = input.shape[axis];
  });
  return sizes;
}

export type Projection = {
  flops: number;
  peakBytes: number;
  /** Tensors whose size could not be projected, kept at their recorded size. */
  kept: number;
};

/**
 * The run's compute and its live-memory peak at other symbol sizes: every
 * tensor resized by its symbolic shape, weights and constants as recorded.
 * An estimate from shapes, as the lenses are.
 */
export function project(
  trace: Trace,
  labelsOf: (tensorId: string) => (string | null)[] | null,
  values: Record<string, number>,
): Projection {
  let kept = 0;
  const tensors: Record<string, Tensor> = {};
  for (const [id, tensor] of Object.entries(trace.tensors)) {
    const labels = labelsOf(id);
    if (!labels || tensor.role === "parameter") {
      tensors[id] = tensor;
      continue;
    }
    const shape = tensor.shape.map((_, axis) => {
      const label = labels[axis];
      if (label === null || label === undefined) return null;
      return evaluateSize(label, values);
    });
    if (shape.some((size) => size === null)) {
      kept++;
      tensors[id] = tensor;
      continue;
    }
    const sizes = shape as number[];
    tensors[id] = {
      ...tensor,
      shape: sizes,
      numel: sizes.reduce((a, b) => a * b, 1),
    };
  }
  const projected = { ...trace, tensors };
  const flops = trace.operations.reduce(
    (sum, op) => sum + stepFlops(op, tensors),
    0,
  );
  return {
    flops,
    peakBytes: liveMemory(projected).peak?.bytes ?? 0,
    kept,
  };
}

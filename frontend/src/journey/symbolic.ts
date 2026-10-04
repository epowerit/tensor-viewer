import type { Run } from "../api/client";
import { matchRuns } from "./reload";

type Trace = Run["trace"];

/** A shape check of the same code with one input axis doubled. */
export type Sweep = {
  /** The input axis that was doubled, and its symbol: `T`, `B`. */
  symbol: string;
  /** Its size in the run being explained. */
  size: number;
  trace: Trace;
};

const SYMBOLS: [RegExp, string][] = [
  [/^batch|^n$|^samples?$/i, "B"],
  [/token|seq|time|length|positions?|steps?|words?/i, "T"],
  [/channel/i, "C"],
  [/height|rows?$/i, "H"],
  [/width|columns?$/i, "W"],
  [/feature|hidden|dim|embed/i, "D"],
];

/** A short symbol for an input axis: `B` for batch, `T` for tokens. */
export function axisSymbol(name: string | null | undefined, axis: number) {
  const found = SYMBOLS.find(([pattern]) => pattern.test(name ?? ""));
  if (found) return found[1];
  const first = (name ?? "").match(/[a-z]/i)?.[0];
  return first ? first.toUpperCase() : `N${axis}`;
}

/** A rational as a coefficient: 3, 1/2, or null when neither. */
function coefficient(value: number): { times: number; over: number } | null {
  if (Number.isInteger(value)) return { times: value, over: 1 };
  const inverse = 1 / value;
  return Math.abs(inverse - Math.round(inverse)) < 1e-9
    ? { times: 1, over: Math.round(inverse) }
    : null;
}

/** `T`, `3·T`, `T/2`, `T+1`, `2·T−2`. */
function linear(symbol: string, slope: number, offset: number) {
  const at = coefficient(slope);
  if (!at || !Number.isInteger(offset)) return null;
  const term =
    at.over > 1
      ? `${symbol}/${at.over}`
      : at.times === 1
        ? symbol
        : `${at.times}·${symbol}`;
  return offset === 0
    ? term
    : `${term}${offset > 0 ? "+" : "−"}${Math.abs(offset)}`;
}

/**
 * Each result's shape in terms of the input's axes, from shape checks that
 * doubled one input axis each: a size that doubles with the tokens is `T`,
 * one that grows by T·k is `k·T`, one that doubles with either of two axes
 * their product (`B·T`), one that quadruples when T doubles `T²`, and one
 * that never moves stays a number. A size that changes in any other way is
 * left as recorded (null). Results are keyed by tensor id in the base run.
 */
export function symbolicShapes(
  base: Trace,
  sweeps: Sweep[],
): Map<string, (string | null)[]> {
  const shapes = new Map<string, (string | null)[]>();
  const sizes = sweeps.map((sweep) => {
    const match = matchRuns(base, sweep.trace);
    const after = new Map(sweep.trace.operations.map((op) => [op.id, op]));
    return (opId: string, output: number) => {
      const op = after.get(match.operations.get(opId) ?? "");
      return op ? sweep.trace.tensors[op.outputs[output]]?.shape : undefined;
    };
  });
  for (const op of base.operations)
    op.outputs.forEach((tensorId, output) => {
      const tensor = base.tensors[tensorId];
      if (!tensor || shapes.has(tensorId)) return;
      const grown = sizes.map((at) => at(op.id, output));
      shapes.set(
        tensorId,
        tensor.shape.map((size, axis) => {
          const growth = grown.map((shape) =>
            shape && shape.length === tensor.shape.length
              ? shape[axis] - size
              : null,
          );
          if (growth.some((value) => value === null)) return null;
          const moving = growth.flatMap((value, k) => (value ? [k] : []));
          if (!moving.length) return `${size}`;
          if (moving.length === 1) {
            const [k] = moving;
            const { symbol, size: x } = sweeps[k];
            // Quadrupled when the axis doubled: its square.
            if (growth[k] === 3 * size && size % (x * x) === 0) {
              const times = size / (x * x);
              return `${times === 1 ? "" : `${times}·`}${symbol}²`;
            }
            const slope = growth[k]! / x;
            return linear(symbol, slope, size - slope * x);
          }
          // Doubling either of two axes doubles it: their product.
          if (moving.length === 2 && moving.every((k) => growth[k] === size)) {
            const [a, b] = moving.map((k) => sweeps[k]);
            const times = size / (a.size * b.size);
            if (!Number.isInteger(times)) return null;
            return `${times === 1 ? "" : `${times}·`}${a.symbol}·${b.symbol}`;
          }
          return null;
        }),
      );
    });
  return shapes;
}

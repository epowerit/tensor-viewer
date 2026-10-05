import type { Operation } from "../api/client";

export type WarmTimes = {
  /** Each timed step's time in µs, its kind's one-time setup taken out. */
  times: Map<string, number>;
  /** The setup taken out of first calls, in µs. */
  setup: number;
  /** First calls whose time was estimated from their kind's later calls. */
  estimated: Set<string>;
};

const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
};

/**
 * Measured step times without one-time setup. The first call of a kind (the
 * first linear, the first softmax) often takes a hundred times longer than
 * the ones after it, setting up kernels and caches. When a kind runs again,
 * its first call is counted at the median of its later calls, and the rest
 * is reported as setup; a kind that runs once keeps its measured time.
 */
export function warmTimes(operations: Operation[]): WarmTimes {
  const times = new Map<string, number>();
  const estimated = new Set<string>();
  let setup = 0;
  const byKind = new Map<string, Operation[]>();
  for (const op of [...operations].sort((a, b) => a.index - b.index)) {
    if (typeof op.duration_us !== "number") continue;
    times.set(op.id, op.duration_us);
    const calls = byKind.get(op.kind) ?? [];
    calls.push(op);
    byKind.set(op.kind, calls);
  }
  for (const [first, ...later] of byKind.values()) {
    if (!later.length) continue;
    const warm = median(later.map((op) => op.duration_us!));
    const measured = first.duration_us!;
    if (measured > warm) {
      times.set(first.id, warm);
      setup += measured - warm;
      estimated.add(first.id);
    }
  }
  return { times, setup, estimated };
}

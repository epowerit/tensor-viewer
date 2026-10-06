import type { Operation, Run, Tensor } from "../api/client";

/** One attention weights tensor: heads of queries × keys. */
export type AttentionMap = {
  op: Operation;
  tensor: Tensor;
  heads: number;
  queries: number;
  keys: number;
  /** Where in the model: the innermost module path, such as blocks.0.attention. */
  label: string;
};

/**
 * The run's attention weights: softmax results over their last axis that
 * are named queries × keys, or shaped batch × heads × queries × keys.
 */
export function attentionMaps(trace: Run["trace"]): AttentionMap[] {
  const found: AttentionMap[] = [];
  for (const op of trace.operations) {
    if (op.kind !== "softmax" || op.status === "error") continue;
    const tensor = trace.tensors[op.outputs[0]];
    if (!tensor || tensor.shape.length < 2) continue;
    const rank = tensor.shape.length;
    const named =
      tensor.axes[rank - 1] === "keys" && tensor.axes[rank - 2] === "queries";
    if (!named && rank !== 4) continue;
    const dim = op.arguments?.dim;
    if (typeof dim === "number" && dim !== -1 && dim !== rank - 1) continue;
    const label =
      op.module?.split(" / ").at(-1)?.trim() || `step ${op.index + 1}`;
    found.push({
      op,
      tensor,
      heads: rank >= 3 ? tensor.shape[rank - 3] : 1,
      queries: tensor.shape[rank - 2],
      keys: tensor.shape[rank - 1],
      label,
    });
  }
  return found;
}

/** One head's weights for the first item of the batch, row by row. */
export function headWeights(
  values: ArrayLike<number>,
  map: AttentionMap,
  head: number,
): number[][] {
  const size = map.queries * map.keys;
  const start = head * size;
  return Array.from({ length: map.queries }, (_, q) =>
    Array.from(
      { length: map.keys },
      (_, k) => values[start + q * map.keys + k] ?? 0,
    ),
  );
}

/** What a head mostly does, in plain words, and how much weight that takes. */
export type HeadPattern = {
  kind: "itself" | "previous" | "first" | "spread";
  /** The average weight on that pattern; for "spread", on the strongest one. */
  share: number;
  /** The pattern taking the most weight, even when it is not most of it. */
  strongest: Exclude<HeadPattern["kind"], "spread">;
};

/** A head reads as one pattern when that takes at least this much weight. */
const PATTERN = 0.4;

/**
 * A head's pattern from its weights (rows: words reading, columns: words read
 * from): each word looking at itself, at the word just before, or at the first
 * word, on average over the words that have a choice (from the second on, so
 * a causal mask's first row, which can only look at itself, does not count);
 * or spread out when none of these takes most of the weight.
 */
export function headPattern(rows: number[][]): HeadPattern {
  const mean = (values: number[]) =>
    values.length
      ? values.reduce((sum, value) => sum + value, 0) / values.length
      : 0;
  const reading = rows.slice(1);
  const shares = {
    itself: mean(reading.map((row, at) => row[at + 1] ?? 0)),
    previous: mean(reading.map((row, at) => row[at] ?? 0)),
    first: mean(reading.map((row) => row[0] ?? 0)),
  };
  // The word before the second word is the first: count it once, as the
  // previous word, so a previous-word head is not read as a first-word one.
  if (reading.length > 1)
    shares.first = mean(reading.slice(1).map((row) => row[0] ?? 0));
  const [strongest, share] = (
    Object.entries(shares) as [HeadPattern["strongest"], number][]
  ).sort((a, b) => b[1] - a[1])[0];
  return {
    kind: share >= PATTERN ? strongest : "spread",
    share,
    strongest,
  };
}

/** How a pattern reads beside its head. */
export const PATTERN_NAMES: Record<HeadPattern["kind"], string> = {
  itself: "itself",
  previous: "word before",
  first: "first word",
  spread: "spread out",
};

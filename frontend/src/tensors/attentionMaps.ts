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

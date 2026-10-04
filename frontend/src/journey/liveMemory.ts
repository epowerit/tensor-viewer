import type { Run } from "../api/client";
import { tensorBytes } from "../tensors/memory";

type Trace = Run["trace"];

/** One storage of activations: when it is made and when it is last read. */
type Held = {
  /** The first tensor written to it, which names it. */
  tensorId: string;
  bytes: number;
  /** Step indices: made at `from`, needed through `to`. */
  from: number;
  to: number;
};

export type LiveMemory = {
  /** Bytes of activations alive at each step, by operation id. */
  live: Map<string, number>;
  /** The step where the most is alive, and what holds it, largest first. */
  peak: {
    opId: string;
    step: number;
    bytes: number;
    holders: { tensorId: string; bytes: number }[];
  } | null;
};

/**
 * The activation memory alive at each step, if every tensor were freed right
 * after the last step that reads it: the least the model's activations need.
 * Views of one storage count once; weights and their views are left out;
 * inputs are held from the start and the model's results to the end. Eager
 * PyTorch can hold more, while a Python name still refers to a tensor no step
 * reads any more.
 */
export function liveMemory(trace: Trace): LiveMemory {
  const held = new Map<string, Held>();
  const last = trace.operations.length - 1;
  const keyOf = (tensorId: string) =>
    trace.tensors[tensorId]?.storage_id ?? tensorId;
  // Weights, and views of them (a tied head's weight.T), are not activations.
  const weights = new Set(
    Object.values(trace.tensors)
      .filter((tensor) => tensor.role === "parameter")
      .map((tensor) => tensor.storage_id ?? tensor.id),
  );
  const hold = (tensorId: string, at: number) => {
    const tensor = trace.tensors[tensorId];
    if (!tensor || weights.has(keyOf(tensorId))) return;
    const key = keyOf(tensorId);
    const bytes = tensorBytes(tensor) ?? 0;
    const found = held.get(key);
    if (!found) held.set(key, { tensorId, bytes, from: at, to: at });
    else {
      found.bytes = Math.max(found.bytes, bytes);
      found.from = Math.min(found.from, at);
      found.to = Math.max(found.to, at);
    }
  };
  for (const id of trace.input_ids) hold(id, 0);
  trace.operations.forEach((op, at) => {
    for (const id of [
      ...op.outputs,
      ...(op.mutations ?? []).map((m) => m.after),
    ])
      hold(id, at);
    for (const id of [
      ...op.inputs,
      ...(op.mutations ?? []).map((m) => m.before),
    ]) {
      const found = held.get(keyOf(id));
      if (found) found.to = Math.max(found.to, at);
    }
  });
  for (const id of [...trace.input_ids, ...trace.output_ids]) {
    const found = held.get(keyOf(id));
    if (found) found.to = last;
  }
  const live = new Map<string, number>();
  let peak: LiveMemory["peak"] = null;
  trace.operations.forEach((op, at) => {
    const alive = [...held.values()].filter(
      (each) => each.from <= at && at <= each.to,
    );
    const bytes = alive.reduce((sum, each) => sum + each.bytes, 0);
    live.set(op.id, bytes);
    if (!peak || bytes > peak.bytes)
      peak = {
        opId: op.id,
        step: at + 1,
        bytes,
        holders: alive
          .map((each) => ({ tensorId: each.tensorId, bytes: each.bytes }))
          .sort((a, b) => b.bytes - a.bytes),
      };
  });
  return { live, peak };
}

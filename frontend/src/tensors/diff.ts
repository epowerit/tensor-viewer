import { createContext } from "react";
import { api, type ChangeSummary, type Run, type Tensor } from "../api/client";
import { isBroken } from "./find";

/**
 * The same tensor state in an earlier run: the input at the same position,
 * or the output at the same position of the same kind of step at the same
 * index, with the same shape and dtype. Null when nothing lines up.
 */
export function matchingState(
  run: Run["trace"],
  earlier: Run["trace"],
  tensorId: string,
): Tensor | null {
  const now = run.tensors[tensorId];
  if (!now) return null;
  let id: string | undefined;
  const input = run.input_ids.indexOf(tensorId);
  if (input >= 0) id = earlier.input_ids[input];
  else {
    const op = run.operations.find((step) => step.outputs.includes(tensorId));
    const then = op && earlier.operations[op.index];
    if (op && then && then.kind === op.kind)
      id = then.outputs[op.outputs.indexOf(tensorId)];
  }
  const before = id ? earlier.tensors[id] : undefined;
  return before &&
    before.shape.join() === now.shape.join() &&
    before.dtype === now.dtype
    ? (before as Tensor)
    : null;
}

export type CellDeltas = {
  /** Now minus before, per flat index; null where either is not a number. */
  deltas: (number | null)[];
  low: number;
  high: number;
  /** How many values differ, and how many could be compared. */
  changed: number;
  compared: number;
};

/** Per-cell change between two recorded states, when both are inline. */
export function cellDeltas(
  now: Pick<Tensor, "values" | "numel">,
  before: Pick<Tensor, "values" | "numel">,
): CellDeltas | null {
  if (
    now.values.length !== now.numel ||
    before.values.length !== before.numel ||
    now.numel !== before.numel
  )
    return null;
  let low = 0,
    high = 0,
    changed = 0,
    compared = 0;
  const deltas = now.values.map((value, index) => {
    const then = before.values[index];
    if (isBroken(value) || isBroken(then)) {
      // A NaN that stayed NaN is unchanged; one that appeared is a change.
      if (isBroken(value) !== isBroken(then) || String(value) !== String(then))
        changed += 1;
      compared += 1;
      return null;
    }
    const delta = Number(value) - Number(then);
    if (!Number.isFinite(delta)) return null;
    compared += 1;
    if (delta !== 0) changed += 1;
    low = Math.min(low, delta);
    high = Math.max(high, delta);
    return delta;
  });
  return { deltas, low, high, changed, compared };
}

/**
 * The displayed run and the run recorded before it, for grids comparing
 * their values. `request` asks for the earlier run, which loads on demand.
 */
export const RunBeforeContext = createContext<{
  run: Run;
  before: Run | null;
  request: () => void;
} | null>(null);

const runs = new Map<string, Promise<Run>>();

/** A recorded run, fetched once and shared by everything comparing with it. */
export function loadRun(id: string) {
  let found = runs.get(id);
  if (!found) {
    found = api.getRun(id);
    // A failed fetch is not remembered, so a later request tries again.
    found.catch(() => runs.delete(id));
    runs.set(id, found);
  }
  return found;
}

export type StateChange =
  | { kind: "changed"; changed: number; compared: number }
  | { kind: "same" }
  | { kind: "new" }
  /** Values in snapshots: `before` names the earlier state to compare. */
  | { kind: "unknown"; before: string };

/** How a tensor state changed since the run before, for a summary. */
export function stateChange(
  run: Run["trace"],
  earlier: Run["trace"],
  tensorId: string,
): StateChange {
  const before = matchingState(run, earlier, tensorId);
  if (!before) return { kind: "new" };
  const deltas = cellDeltas(run.tensors[tensorId] as Tensor, before);
  if (!deltas) return { kind: "unknown", before: before.id };
  return deltas.changed
    ? { kind: "changed", changed: deltas.changed, compared: deltas.compared }
    : { kind: "same" };
}

const comparisons = new Map<string, Promise<(ChangeSummary | null)[]>>();

/** A backend summary as a state change; null when it could not compare. */
export function summaryChange(
  summary: ChangeSummary | null,
): StateChange | null {
  if (!summary) return null;
  return summary.changed
    ? { kind: "changed", changed: summary.changed, compared: summary.compared }
    : { kind: "same" };
}

/**
 * How tensors changed since the run before, answered by the backend when
 * their values stay in snapshots. Shared and kept, since runs never change.
 */
export function loadComparison(
  runId: string,
  earlierId: string,
  pairs: [string, string][],
) {
  const key = `${runId}/${earlierId}/${pairs.map((pair) => pair.join(":")).join(",")}`;
  let found = comparisons.get(key);
  if (!found) {
    found = api
      .compareRun(runId, earlierId, pairs)
      .then((result) => result.results);
    found.catch(() => comparisons.delete(key));
    comparisons.set(key, found);
  }
  return found;
}

/**
 * How close two recorded states are, when both are inline: the backend's
 * comparison, in the browser. `allclose` follows torch.allclose's defaults
 * (rtol 1e-5, atol 1e-8; NaN is never close, equal infinities are).
 */
export function closeness(
  a: Pick<Tensor, "values" | "numel">,
  b: Pick<Tensor, "values" | "numel">,
): ChangeSummary | null {
  const deltas = cellDeltas(a, b);
  if (!deltas) return null;
  let max = 0,
    total = 0,
    finite = 0,
    allclose = true;
  a.values.forEach((raw, index) => {
    const x = asValue(raw),
      y = asValue(b.values[index]);
    if (Number.isNaN(x) || Number.isNaN(y)) allclose = false;
    else if (!Number.isFinite(x) || !Number.isFinite(y)) {
      if (x !== y) allclose = false;
    } else {
      const gap = Math.abs(x - y);
      max = Math.max(max, gap);
      total += gap;
      finite += 1;
      if (gap > 1e-8 + 1e-5 * Math.abs(y)) allclose = false;
    }
  });
  return {
    changed: deltas.changed,
    compared: deltas.compared,
    low: deltas.low,
    high: deltas.high,
    max_abs: max,
    mean_abs: finite ? total / finite : 0,
    allclose,
  };
}

function asValue(value: number | string | boolean | undefined) {
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return Number(value);
  const word = String(value).trim().toLowerCase();
  return word === "inf" || word === "infinity"
    ? Infinity
    : word === "-inf" || word === "-infinity"
      ? -Infinity
      : Number(value);
}

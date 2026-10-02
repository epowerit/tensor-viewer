import type { Run, Tensor } from "../api/client";

export type StepInfo = {
  id: string;
  kind: string;
  name: string | null;
  shape: number[] | null;
  dtype: string | null;
  outputs: (Tensor | undefined)[];
  failed: boolean;
  line: number | null;
  text: string;
};
export type StepDiff = {
  index: number;
  left: StepInfo | null;
  right: StepInfo | null;
  change:
    | "same"
    | "values"
    | "shape"
    | "outputs"
    | "operation"
    | "status"
    | "added"
    | "removed";
  /** Largest absolute difference between recorded values, when comparable. */
  delta: number | null;
  /** Null when structure differs or there are no output tensors to compare. */
  valuesCompared: boolean | null;
  /**
   * Means of a large tensor whose values were not sent, when its recorded
   * distributions differ: evidence of a change without the values.
   */
  shift?: { before: number; after: number } | null;
};

function step(run: Run, index: number): StepInfo | null {
  const op = run.trace.operations[index];
  if (!op) return null;
  const output = run.trace.tensors[op.outputs[0]];
  return {
    id: op.id,
    kind: op.kind,
    name: output?.name ?? null,
    shape: output?.shape ?? null,
    dtype: output?.dtype ?? null,
    outputs: op.outputs.map((id) => run.trace.tensors[id]),
    failed: op.status === "error",
    line: op.source?.line ?? null,
    text: op.source?.text ?? "",
  };
}

/** Compare every aligned output; unavailable values are not evidence of equality. */
function compareValues(a: StepInfo, b: StepInfo) {
  let complete = true;
  let different = false;
  let numeric = true;
  let delta = 0;
  let shift: StepDiff["shift"] = null;
  for (let output = 0; output < a.outputs.length; output++) {
    const left = a.outputs[output],
      right = b.outputs[output];
    // Equal values always give equal histograms, so differing histograms
    // prove a change even when the values themselves were not sent.
    if (
      left?.histogram &&
      right?.histogram &&
      (left.value_source === "paged" || right.value_source === "paged") &&
      JSON.stringify(left.histogram) !== JSON.stringify(right.histogram)
    ) {
      different = true;
      if (
        !shift &&
        typeof left.histogram.mean === "number" &&
        typeof right.histogram.mean === "number"
      )
        shift = { before: left.histogram.mean, after: right.histogram.mean };
    }
    if (
      !left ||
      !right ||
      left.value_source === "shape" ||
      right.value_source === "shape" ||
      left.value_source === "paged" ||
      right.value_source === "paged" ||
      left.values.length !== left.numel ||
      right.values.length !== right.numel ||
      left.numel !== right.numel
    ) {
      complete = false;
      continue;
    }
    for (let i = 0; i < left.numel; i++) {
      const x = left.values[i],
        y = right.values[i];
      if (x === y) continue;
      different = true;
      // Strings also preserve exact integers outside JavaScript's safe range.
      // Record a mismatch without converting them to an imprecise number.
      if (typeof x !== "number" || typeof y !== "number") numeric = false;
      else delta = Math.max(delta, Math.abs(x - y));
    }
  }
  return {
    different,
    complete,
    delta: complete && numeric ? delta : null,
    shift,
  };
}

/** Align two runs step by step in execution order. */
export function compareRuns(left: Run, right: Run): StepDiff[] {
  const count = Math.max(
    left.trace.operations.length,
    right.trace.operations.length,
  );
  return Array.from({ length: count }, (_, index) => {
    const a = step(left, index),
      b = step(right, index);
    if (!a || !b)
      return {
        index,
        left: a,
        right: b,
        change: a ? "removed" : "added",
        delta: null,
        valuesCompared: null,
      };
    let change: StepDiff["change"] = "same";
    let delta: number | null = null;
    let valuesCompared: boolean | null = null;
    let shift: StepDiff["shift"] = null;
    if (a.kind !== b.kind) change = "operation";
    else if (a.failed !== b.failed) change = "status";
    else if (a.outputs.length !== b.outputs.length) change = "outputs";
    else if (
      a.outputs.some(
        (output, i) =>
          output?.shape.join() !== b.outputs[i]?.shape.join() ||
          output?.dtype !== b.outputs[i]?.dtype,
      )
    )
      change = "shape";
    else if (a.outputs.length) {
      const values = compareValues(a, b);
      delta = values.delta;
      valuesCompared = values.complete;
      shift = values.shift;
      if (values.different) change = "values";
    }
    return { index, left: a, right: b, change, delta, valuesCompared, shift };
  });
}

export function summarize(steps: StepDiff[]): string {
  const changed = steps.filter((item) => item.change !== "same");
  const unavailable = steps.filter(
    (item) => item.valuesCompared === false,
  ).length;
  const evidence = unavailable
    ? ` Values were not fully compared for ${unavailable} ${unavailable === 1 ? "step" : "steps"}.`
    : "";
  if (!steps.length) return "Neither run recorded an operation.";
  if (!changed.length) {
    const values =
      !unavailable && steps.some((item) => item.valuesCompared === true)
        ? ", and recorded values"
        : "";
    return `All ${steps.length} steps match in operation, status, output shapes and types${values}.${evidence}`;
  }
  const first = changed[0];
  return `${changed.length} of ${steps.length} steps differ, starting at step ${first.index + 1}.${evidence}`;
}

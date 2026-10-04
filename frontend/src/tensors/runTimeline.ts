import { createContext } from "react";
import type { Run, RunSummary, Tensor } from "../api/client";
import { matchRuns } from "../journey/reload";

type Trace = Run["trace"];

/** One run's version of a tensor. */
export type RunPoint = {
  runId: string;
  createdAt: string;
  tensor: Tensor;
  /** Its shape differs from the run before it. */
  reshaped: boolean;
};

/** Where a tensor sits in a trace: an input by name, or a step's result. */
type Place =
  { input: string } | { op: string; output: number; mutation?: number };

function placeOf(trace: Trace, tensorId: string): Place | null {
  if (trace.input_ids.includes(tensorId)) {
    const name = trace.tensors[tensorId]?.name;
    return name ? { input: name } : null;
  }
  for (const op of trace.operations) {
    const output = op.outputs.indexOf(tensorId);
    if (output >= 0) return { op: op.id, output };
    const mutation = (op.mutations ?? []).findIndex(
      (each) => each.after === tensorId,
    );
    if (mutation >= 0) return { op: op.id, output: 0, mutation };
  }
  return null;
}

function tensorAt(trace: Trace, place: Place): Tensor | undefined {
  if ("input" in place)
    return trace.input_ids
      .map((id) => trace.tensors[id])
      .find((tensor) => tensor?.name === place.input);
  const op = trace.operations.find((each) => each.id === place.op);
  if (!op) return undefined;
  const id =
    place.mutation !== undefined
      ? op.mutations?.[place.mutation]?.after
      : op.outputs[place.output];
  return id ? trace.tensors[id] : undefined;
}

/**
 * One tensor across a project's runs, newest first: each run is lined up
 * with the run after it, as hot reload lines them up, so the tensor is
 * followed through edits that add, remove, or move steps around it. The
 * chain stops at the first run where the step no longer existed.
 */
export function tensorAcrossRuns(
  runs: { id: string; created_at: string; trace: Trace }[],
  tensorId: string,
): RunPoint[] {
  const [current] = runs;
  if (!current) return [];
  let place = placeOf(current.trace, tensorId);
  const points: RunPoint[] = [];
  for (const [index, run] of runs.entries()) {
    if (!place) break;
    const tensor = tensorAt(run.trace, place);
    if (!tensor) break;
    points.push({
      runId: run.id,
      createdAt: run.created_at,
      tensor,
      reshaped: false,
    });
    const older = runs[index + 1];
    if (!older || "input" in place) continue;
    // The older run's step that became this one.
    const { operations } = matchRuns(older.trace, run.trace);
    const opId = place.op;
    const before = [...operations].find(([, after]) => after === opId)?.[0];
    place = before ? { ...place, op: before } : null;
  }
  for (const [index, point] of points.entries()) {
    const older = points[index + 1];
    point.reshaped =
      !!older && older.tensor.shape.join() !== point.tensor.shape.join();
  }
  return points;
}

/** What the tensor details need to read a tensor across the project's runs. */
export type RunHistory = {
  run: Run;
  /** The project's runs, newest first. */
  history: RunSummary[];
  /** Opens a run beside this one in Run compare. */
  compare: (runId: string) => void;
};

export const RunHistoryContext = createContext<RunHistory | null>(null);

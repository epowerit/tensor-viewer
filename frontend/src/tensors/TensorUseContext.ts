import { createContext } from "react";
import type { Run } from "../api/client";

export type UseStep = { id: string; step: number; kind: string };
export type TensorUse = {
  /** The step that produced this state; null for an input or a parameter. */
  made: UseStep | null;
  /** Steps that read this state, in order. */
  read: UseStep[];
};

/**
 * Which step made each recorded tensor state and which steps read it. A write
 * in place makes the new state and reads the old one.
 */
export function tensorUses(trace: Run["trace"]) {
  const made = new Map<string, UseStep>();
  const read = new Map<string, UseStep[]>();
  const reads = (id: string, step: UseStep) => {
    const list = read.get(id) ?? [];
    if (!list.some((other) => other.id === step.id)) list.push(step);
    read.set(id, list);
  };
  for (const op of trace.operations) {
    const step = { id: op.id, step: op.index + 1, kind: op.kind };
    op.inputs?.forEach((id) => reads(id, step));
    op.outputs.forEach((id) => made.set(id, step));
    op.mutations?.forEach((mutation) => {
      reads(mutation.before, step);
      made.set(mutation.after, step);
    });
  }
  return (tensorId: string): TensorUse => ({
    made: made.get(tensorId) ?? null,
    read: read.get(tensorId) ?? [],
  });
}

/** Where the displayed run's tensors come from and go, with a way to go there. */
export const TensorUseContext = createContext<{
  uses: (tensorId: string) => TensorUse;
  /** Opens a step; with a tensor and a cell, that cell selected. */
  go: (stepId: string, tensorId?: string, cell?: number) => void;
  /** The displayed run's trace, for comparing one tensor with another. */
  trace?: Run["trace"];
  runId?: string;
} | null>(null);

import { useSyncExternalStore } from "react";
import { api, type Dependencies } from "../api/client";
import type { DependencyAsk } from "./dependencies";

/**
 * The last who-sees-whom check asked about each run, kept outside the tab so
 * that a breach stays among the run notes after the tab is closed. Answers
 * are kept per question, so asking again shows the answer at once.
 */
export type DependencyCheck = {
  runId: string;
  ask: DependencyAsk;
  answer?: Dependencies;
  error?: string;
};

const listeners = new Set<() => void>();
const latest = new Map<string, DependencyCheck>();
const answers = new Map<string, Promise<Dependencies>>();

const keyOf = (runId: string, ask: DependencyAsk) =>
  `${runId}|${ask.tensor_id}|${ask.axis}|${ask.input_axis}|${ask.expect}`;
const notify = () => listeners.forEach((listener) => listener());

/** Ask a run who sees whom; the answer replaces the run's last check. */
export function askDependencies(runId: string, ask: DependencyAsk) {
  const key = keyOf(runId, ask);
  let asked = answers.get(key);
  if (!asked) {
    asked = api.dependencies(runId, {
      tensor_id: ask.tensor_id,
      axis: ask.axis,
      input_axis: ask.input_axis,
      expect: ask.expect,
    });
    asked.catch(() => answers.delete(key));
    answers.set(key, asked);
  }
  const check: DependencyCheck = { runId, ask };
  latest.set(runId, check);
  notify();
  const settle = (update: Partial<DependencyCheck>) => {
    // A later question about the run wins over this one.
    if (latest.get(runId) !== check) return;
    latest.set(runId, { ...check, ...update });
    notify();
  };
  asked.then(
    (answer) => settle({ answer }),
    (error: Error) => settle({ error: error.message }),
  );
}

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

/** The last check asked about a run, answered or not. */
export function useDependencyCheck(
  runId: string | null | undefined,
): DependencyCheck | null {
  return useSyncExternalStore(subscribe, () =>
    runId ? (latest.get(runId) ?? null) : null,
  );
}

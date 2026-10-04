import { useEffect, useState, useSyncExternalStore } from "react";
import { api, type Run, type WatchSeries } from "../api/client";

/**
 * What the Watch tab keeps outside itself: which watches pause playback, per
 * project, and each watch's series across a run, fetched once. Playback reads
 * the pauses even while the tab is closed.
 */

const key = (projectId: string) => `tensorviewer.pause.${projectId}`;
const listeners = new Set<() => void>();
const pauses = new Map<string, string[]>();

function read(projectId: string): string[] {
  if (!pauses.has(projectId)) {
    let saved: unknown = [];
    try {
      saved = JSON.parse(localStorage.getItem(key(projectId)) ?? "[]");
    } catch {
      saved = [];
    }
    pauses.set(
      projectId,
      Array.isArray(saved)
        ? saved.filter((item): item is string => typeof item === "string")
        : [],
    );
  }
  return pauses.get(projectId)!;
}

export function setPauses(projectId: string, next: string[]) {
  pauses.set(projectId, next);
  try {
    localStorage.setItem(key(projectId), JSON.stringify(next));
  } catch {
    // Storage can be unavailable; the pauses still apply for this visit.
  }
  for (const listener of listeners) listener();
}

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};
const none: string[] = [];

/** The watches that pause playback in a project. */
export function usePauses(projectId: string | null): string[] {
  return useSyncExternalStore(subscribe, () =>
    projectId ? read(projectId) : none,
  );
}

const series = new Map<string, Promise<WatchSeries>>();

/** A watch across a run, asked for once per run and expression. */
export function loadSeries(runId: string, expression: string) {
  const id = `${runId}\n${expression}`;
  let found = series.get(id);
  if (!found) {
    found = api.evaluateSeries(runId, expression).catch(
      (error: Error) =>
        ({
          points: [],
          truncated: false,
          error: error.message,
        }) as WatchSeries,
    );
    series.set(id, found);
  }
  return found;
}

/** The recorded steps where a series is true (non-zero). */
export const holdingSteps = (run: Run, found: WatchSeries) => {
  const steps = new Set(run.trace.operations.map((op) => op.id));
  return (found.points ?? []).filter(
    (point) => !!point.value && steps.has(point.at),
  );
};

/** Steps where any pausing watch holds: breakpoints for playback. */
export function useWatchBreaks(
  run: Run | null,
  projectId: string | null,
): ReadonlySet<string> {
  const pausing = usePauses(projectId);
  const [steps, setSteps] = useState<ReadonlySet<string>>(new Set());
  const ask = `${run?.id}\n${pausing.join("\n")}`;
  useEffect(() => {
    if (!run || !pausing.length) {
      setSteps(new Set());
      return;
    }
    let live = true;
    Promise.all(
      pausing.map((expression) => loadSeries(run.id, expression)),
    ).then((all) => {
      if (!live) return;
      setSteps(
        new Set(
          all.flatMap((found) => holdingSteps(run, found).map((p) => p.at)),
        ),
      );
    });
    return () => {
      live = false;
    };
    // `ask` names the run and the pausing watches.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ask]);
  return steps;
}

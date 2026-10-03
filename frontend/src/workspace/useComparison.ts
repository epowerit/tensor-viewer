import { useEffect, useMemo, useState } from "react";
import type { ChangeSummary, Run } from "../api/client";
import { loadComparison } from "../tensors/diff";
import {
  compareRuns,
  settleWithBackend,
  snapshotPairs,
  type StepDiff,
} from "./compare";

/**
 * Two runs aligned step by step, with the tensors too large to send compared
 * by the backend over their snapshots, in one request. `pending` is true
 * while those answers are on their way.
 */
export function useComparison(
  current: Run | null,
  other: Run | null,
): { steps: StepDiff[] | null; pending: boolean } {
  const aligned = useMemo(
    () => (current && other ? compareRuns(current, other) : null),
    [current, other],
  );
  const pairs = useMemo(
    () => (aligned ? snapshotPairs(aligned) : []),
    [aligned],
  );
  const key =
    current && other && pairs.length ? `${current.id}|${other.id}` : null;
  const [answers, setAnswers] = useState<{
    key: string | null;
    found: Map<string, ChangeSummary | null>;
  }>({ key: null, found: new Map() });
  useEffect(() => {
    if (!key || !current || !other) return;
    let live = true;
    loadComparison(current.id, other.id, pairs)
      .then((results) => {
        if (live)
          setAnswers({
            key,
            found: new Map(pairs.map(([a, b], i) => [`${a}|${b}`, results[i]])),
          });
      })
      .catch(() => live && setAnswers({ key, found: new Map() }));
    return () => {
      live = false;
    };
    // `key` and `pairs` follow the two runs.
  }, [key, pairs, current, other]);
  const steps = useMemo(
    () =>
      aligned &&
      (answers.key === key
        ? settleWithBackend(aligned, answers.found)
        : aligned),
    [aligned, answers, key],
  );
  return { steps, pending: !!key && answers.key !== key };
}

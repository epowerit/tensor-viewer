import { useEffect, useState } from "react";
import { api, type ReductionStatistics } from "../api/client";
import {
  createReductionLoader,
  reductionKey,
  type ReductionGroup,
} from "./reductionStatistics";

const load = createReductionLoader((group, signal) =>
  api.reductionStatistics(
    group.runId,
    group.operationId,
    group.outputIndex,
    signal,
  ),
);

export function useReductionStatistics(
  group: ReductionGroup,
  enabled: boolean,
) {
  const [attempt, setAttempt] = useState(0);
  const key = `${reductionKey(group)}/${attempt}`;
  const [state, setState] = useState<{
    key: string;
    data: ReductionStatistics | null;
    error: string;
  }>({ key: "", data: null, error: "" });
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    load(group, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setState({ key, data, error: "" });
      })
      .catch((error) => {
        if (!controller.signal.aborted)
          setState({ key, data: null, error: error.message });
      });
    return () => controller.abort();
  }, [key, enabled]);
  return {
    data: enabled && state.key === key ? state.data : null,
    loading: enabled && state.key !== key,
    error: enabled && state.key === key ? state.error : "",
    retry: () => setAttempt((n) => n + 1),
  };
}

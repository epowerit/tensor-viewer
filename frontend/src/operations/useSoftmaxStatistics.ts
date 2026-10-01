import { useEffect, useState } from "react";
import { api, type SoftmaxStatistics } from "../api/client";
import {
  createSoftmaxLoader,
  softmaxKey,
  type SoftmaxGroup,
} from "./softmaxStatistics";
const load = createSoftmaxLoader((p, signal) =>
  api.softmaxStatistics(p.runId, p.operationId, p.group, signal),
);
export function useSoftmaxStatistics(group: SoftmaxGroup, enabled: boolean) {
  const [attempt, setAttempt] = useState(0);
  const key = `${softmaxKey(group)}/${attempt}`;
  const [state, setState] = useState<{
    key: string;
    data: SoftmaxStatistics | null;
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

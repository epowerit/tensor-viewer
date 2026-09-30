import { useEffect, useState } from "react";
import { api, type NormalizationStatistics } from "../api/client";
import {
  createNormalizationLoader,
  normalizationKey,
  type NormalizationGroup,
} from "./normalizationStatistics";

const load = createNormalizationLoader((group, signal) =>
  api.normalizationStatistics(
    group.runId,
    group.operationId,
    group.group,
    signal,
  ),
);

export function useNormalizationStatistics(
  group: NormalizationGroup,
  enabled: boolean,
) {
  const [attempt, setAttempt] = useState(0);
  const key = `${normalizationKey(group)}/${attempt}`;
  const [state, setState] = useState<{
    key: string;
    data: NormalizationStatistics | null;
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

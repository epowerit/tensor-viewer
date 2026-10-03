import { useEffect, useState } from "react";
import { api } from "../api/client";

type Question =
  | "search"
  | "landmarks"
  | "margins"
  | "region"
  | "axis"
  | "thumbnails"
  | "sums";

/**
 * Ask the backend a whole-tensor question about a recorded tensor whose
 * values stay in a snapshot. `params` null asks nothing; `delay` waits for
 * typing to pause before asking.
 */
export function useTensorQuery<T>(
  runId: string | undefined,
  tensorId: string,
  question: Question,
  params: Record<string, string | number | boolean> | null,
  delay = 0,
): { data: T | null; loading: boolean; error: string } {
  const key = params && runId ? JSON.stringify(params) : null;
  const [state, setState] = useState<{
    key: string | null;
    data: T | null;
    error: string;
  }>({ key: null, data: null, error: "" });
  useEffect(() => {
    if (!key || !runId) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      api
        .tensorQuery<T>(
          runId,
          tensorId,
          question,
          JSON.parse(key),
          controller.signal,
        )
        .then((data) => setState({ key, data, error: "" }))
        .catch((error: Error) => {
          if (error.name !== "AbortError")
            setState({ key, data: null, error: error.message });
        });
    }, delay);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [key, runId, tensorId, question, delay]);
  if (!key) return { data: null, loading: false, error: "" };
  return {
    data: state.key === key ? state.data : null,
    loading: state.key !== key,
    error: state.key === key ? state.error : "",
  };
}

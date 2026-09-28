import { useEffect, useState } from "react";
import { api, type Tensor } from "../api/client";

/** A bounded window is fetched from an immutable snapshot, never the live model. */
export function useTensorValues(
  tensor: Tensor,
  runId: string | undefined,
  indices: number[],
) {
  const key = `${runId}/${tensor.id}/${indices.join(",")}`;
  const [retry, setRetry] = useState(0);
  const [state, setState] = useState<{
    key: string;
    values: Record<number, number | string>;
    error: string;
  }>({ key: "", values: {}, error: "" });
  const paged = tensor.value_source === "paged";
  useEffect(() => {
    if (!paged || !runId || !indices.length) return;
    const controller = new AbortController();
    api
      .tensorValues(runId, tensor.id, indices, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted)
          setState({
            key,
            values: Object.fromEntries(
              result.indices.map((index, i) => [index, result.values[i]]),
            ),
            error: "",
          });
      })
      .catch((error) => {
        if (!controller.signal.aborted)
          setState({ key, values: {}, error: error.message });
      });
    return () => controller.abort();
  }, [key, paged, retry]);
  return {
    valueAt: (index: number) =>
      paged
        ? state.key === key
          ? state.values[index]
          : undefined
        : tensor.values[index],
    loading: paged && state.key !== key,
    error: state.key === key ? state.error : "",
    retry: () => setRetry((value) => value + 1),
  };
}

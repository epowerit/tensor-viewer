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
    // Detailed 3D views can show up to 1,024 cells. Keep every snapshot
    // request within the API's 256-index bound and cancel the whole window.
    const chunks = Array.from(
      { length: Math.ceil(indices.length / 256) },
      (_, i) => indices.slice(i * 256, (i + 1) * 256),
    );
    Promise.all(
      chunks.map((chunk) =>
        api.tensorValues(runId, tensor.id, chunk, controller.signal),
      ),
    )
      .then((results) => {
        if (!controller.signal.aborted)
          setState({
            key,
            values: Object.fromEntries(
              results.flatMap((result) =>
                result.indices.map((index, i) => [index, result.values[i]]),
              ),
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

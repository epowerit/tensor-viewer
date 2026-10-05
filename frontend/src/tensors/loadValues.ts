import type { Tensor } from "../api/client";
import { readNpy } from "./npy";

/** Larger tensors are not fetched whole. */
export const WHOLE = 400_000;

const loaded = new Map<string, Promise<Float64Array | null>>();

/**
 * A tensor's recorded values, all of them: inline, or its snapshot read as a
 * .npy. Null for a shapes-only state or one larger than WHOLE values.
 */
export function valuesOf(
  runId: string,
  tensor: Tensor,
): Promise<Float64Array | null> {
  if (
    tensor.value_source === "inline" &&
    tensor.values?.length === tensor.numel
  )
    return Promise.resolve(Float64Array.from(tensor.values, Number));
  if (tensor.value_source === "shape" || tensor.numel > WHOLE)
    return Promise.resolve(null);
  const key = `${runId}/${tensor.id}`;
  let asked = loaded.get(key);
  if (!asked) {
    asked = fetch(`/api/v1/runs/${runId}/tensors/${tensor.id}/npy`)
      .then((response) => (response.ok ? response.arrayBuffer() : null))
      .then((buffer) => (buffer ? (readNpy(buffer)?.values ?? null) : null))
      .catch(() => null);
    loaded.set(key, asked);
  }
  return asked;
}

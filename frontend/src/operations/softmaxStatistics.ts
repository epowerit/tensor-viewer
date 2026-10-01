import type { SoftmaxStatistics } from "../api/client";
export type SoftmaxGroup = {
  runId: string;
  operationId: string;
  tensorId: string;
  group: number;
  size: number;
};
export const softmaxKey = (p: SoftmaxGroup) =>
  JSON.stringify([p.runId, p.operationId, p.tensorId, p.group, p.size]);

export function validSoftmaxStatistics(
  data: SoftmaxStatistics,
  p: SoftmaxGroup,
) {
  if (
    data.run_id !== p.runId ||
    data.operation_id !== p.operationId ||
    data.tensor_id !== p.tensorId ||
    data.group !== p.group ||
    data.count !== p.size
  )
    return false;
  if (data.status === "non_finite")
    return (
      data.maximum == null &&
      data.denominator == null &&
      data.masked_count == null
    );
  if (data.status === "all_masked")
    return (
      data.maximum == null &&
      data.denominator == null &&
      data.masked_count === p.size
    );
  return (
    data.status === "ok" &&
    typeof data.maximum === "number" &&
    Number.isFinite(data.maximum) &&
    typeof data.denominator === "number" &&
    Number.isFinite(data.denominator) &&
    data.denominator >= 1 &&
    typeof data.masked_count === "number" &&
    Number.isSafeInteger(data.masked_count) &&
    data.masked_count >= 0 &&
    data.masked_count < p.size &&
    data.denominator <= p.size - data.masked_count
  );
}

export function createSoftmaxLoader(
  fetchGroup: (
    p: SoftmaxGroup,
    signal: AbortSignal,
  ) => Promise<SoftmaxStatistics>,
  capacity = 32,
) {
  const cache = new Map<string, SoftmaxStatistics>();
  return async (p: SoftmaxGroup, signal: AbortSignal) => {
    signal.throwIfAborted();
    const key = softmaxKey(p);
    let data = cache.get(key);
    if (!data) {
      data = await fetchGroup(p, signal);
      signal.throwIfAborted();
      if (!validSoftmaxStatistics(data, p))
        throw new Error(
          "The statistics do not match this complete softmax group.",
        );
    }
    cache.delete(key);
    cache.set(key, data);
    if (cache.size > capacity) cache.delete(cache.keys().next().value!);
    return data;
  };
}

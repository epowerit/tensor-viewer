import type { NormalizationStatistics } from "../api/client";

export type NormalizationGroup = {
  runId: string;
  operationId: string;
  tensorId: string;
  group: number;
  size: number;
};

export const normalizationKey = (group: NormalizationGroup) =>
  JSON.stringify([
    group.runId,
    group.operationId,
    group.tensorId,
    group.group,
    group.size,
  ]);

export function normalizationNumbers(data?: NormalizationStatistics | null) {
  if (
    data?.status !== "ok" ||
    typeof data.mean !== "number" ||
    typeof data.variance !== "number" ||
    typeof data.denominator !== "number" ||
    ![data.mean, data.variance, data.denominator].every(Number.isFinite) ||
    data.variance < 0 ||
    data.denominator < 0
  )
    return null;
  return {
    mean: data.mean,
    variance: data.variance,
    denominator: data.denominator,
  };
}

/** Immutable summaries have a small LRU cache; cancelled requests never populate it. */
export function createNormalizationLoader(
  fetchGroup: (
    group: NormalizationGroup,
    signal: AbortSignal,
  ) => Promise<NormalizationStatistics>,
  capacity = 32,
) {
  const cache = new Map<string, NormalizationStatistics>();
  return async (group: NormalizationGroup, signal: AbortSignal) => {
    signal.throwIfAborted();
    const key = normalizationKey(group);
    let data = cache.get(key);
    if (!data) {
      data = await fetchGroup(group, signal);
      signal.throwIfAborted();
      if (
        data.operation_id !== group.operationId ||
        data.tensor_id !== group.tensorId ||
        data.group !== group.group ||
        data.start !== group.group * group.size ||
        data.count !== group.size ||
        (data.status === "ok"
          ? !normalizationNumbers(data)
          : !["non_finite", "overflow"].includes(data.status) ||
            [data.mean, data.variance, data.denominator].some((n) => n != null))
      )
        throw new Error(
          "The statistics do not match this complete normalization group.",
        );
    }
    cache.delete(key);
    cache.set(key, data);
    if (cache.size > capacity) cache.delete(cache.keys().next().value!);
    return data;
  };
}

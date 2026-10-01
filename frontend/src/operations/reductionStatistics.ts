import type { ReductionStatistics } from "../api/client";

export type ReductionGroup = {
  runId: string;
  operationId: string;
  tensorId: string;
  outputIndex: number;
  size: number;
  kind: "mean" | "sum";
  integer: boolean;
};

export const reductionKey = (group: ReductionGroup) =>
  JSON.stringify([
    group.runId,
    group.operationId,
    group.tensorId,
    group.outputIndex,
    group.size,
    group.kind,
    group.integer,
  ]);

function validValue(value: unknown, integer: boolean) {
  return typeof value === "number"
    ? integer
      ? Number.isSafeInteger(value)
      : Number.isFinite(value)
    : integer &&
        typeof value === "string" &&
        /^-?(0|[1-9]\d{0,26})$/.test(value);
}

export function reductionNumbers(data?: ReductionStatistics | null) {
  if (data?.status !== "ok" || data.result == null) return null;
  return { sum: data.sum ?? undefined, result: data.result };
}

/** Verify full-group identity, never accept a partial window or a stale run. */
export function validReductionStatistics(
  data: ReductionStatistics,
  group: ReductionGroup,
) {
  if (
    data.run_id !== group.runId ||
    data.operation_id !== group.operationId ||
    data.tensor_id !== group.tensorId ||
    data.output_index !== group.outputIndex ||
    data.count !== group.size
  )
    return false;
  if (data.status !== "ok")
    return (
      ["non_finite", "overflow"].includes(data.status) &&
      data.sum == null &&
      data.result == null
    );
  if (!validValue(data.result, group.integer)) return false;
  if (data.sum == null) return group.kind === "mean" && !group.integer;
  if (!validValue(data.sum, group.integer)) return false;
  return group.kind !== "sum" || data.sum === data.result;
}

/** Revisit a group without scanning again. Term/window changes reuse the same key. */
export function createReductionLoader(
  fetchGroup: (
    group: ReductionGroup,
    signal: AbortSignal,
  ) => Promise<ReductionStatistics>,
  capacity = 32,
) {
  const cache = new Map<string, ReductionStatistics>();
  return async (group: ReductionGroup, signal: AbortSignal) => {
    signal.throwIfAborted();
    const key = reductionKey(group);
    let data = cache.get(key);
    if (!data) {
      data = await fetchGroup(group, signal);
      signal.throwIfAborted();
      if (!validReductionStatistics(data, group))
        throw new Error(
          "The calculation does not match this complete reduction group.",
        );
    }
    cache.delete(key);
    cache.set(key, data);
    if (cache.size > capacity) cache.delete(cache.keys().next().value!);
    return data;
  };
}

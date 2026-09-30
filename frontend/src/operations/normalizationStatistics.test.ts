import { describe, expect, it, vi } from "vitest";
import type { NormalizationStatistics } from "../api/client";
import { layerNormCalculation } from "./layerNormalization";
import {
  createNormalizationLoader,
  normalizationNumbers,
  type NormalizationGroup,
} from "./normalizationStatistics";

const group: NormalizationGroup = {
  runId: "run-1",
  operationId: "op-1",
  tensorId: "tensor-1",
  group: 0,
  size: 768,
};
const result = (request = group): NormalizationStatistics => ({
  operation_id: request.operationId,
  tensor_id: request.tensorId,
  group: request.group,
  start: request.group * request.size,
  count: request.size,
  status: "ok",
  mean: 383.5 + request.group * 768,
  variance: (768 ** 2 - 1) / 12,
  denominator: Math.sqrt((768 ** 2 - 1) / 12 + 1e-5),
});
const signal = () => new AbortController().signal;

describe("complete normalization group summaries", () => {
  it("uses every group element rather than a visible eight-cell average", async () => {
    const load = createNormalizationLoader(async (request) => result(request));
    const stats = normalizationNumbers(await load(group, signal()))!;
    expect(stats.mean).toBe(383.5);
    expect(stats.variance).toBe(49151.916666666664);
    const value = layerNormCalculation(stats, 767, 1, 0)!;
    // Native float64 torch.nn.functional.layer_norm(arange(768), (768,))[-1].
    expect(value.output).toBeCloseTo(1.7297969992715976, 10);
    expect(stats.mean).not.toBe((760 + 767) / 2);
  });

  it("caches an immutable group but keeps operations and runs independent", async () => {
    const fetch = vi.fn(async (request: NormalizationGroup) => result(request));
    const load = createNormalizationLoader(fetch);
    for (let i = 0; i < 5; i++) await load(group, signal());
    expect(fetch).toHaveBeenCalledTimes(1);
    for (const changed of [
      { group: 1 },
      { runId: "run-2" },
      { operationId: "op-2" },
      { tensorId: "tensor-2" },
    ])
      await load({ ...group, ...changed }, signal());
    expect(fetch).toHaveBeenCalledTimes(5);
  });

  it("evicts the least recently visited group at the cache bound", async () => {
    const fetch = vi.fn(async (request: NormalizationGroup) => result(request));
    const load = createNormalizationLoader(fetch, 2);
    for (const index of [0, 1, 0, 2, 0])
      await load({ ...group, group: index }, signal());
    expect(fetch).toHaveBeenCalledTimes(3);
    await load({ ...group, group: 1 }, signal());
    expect(fetch).toHaveBeenCalledTimes(4);
  });

  it("discards a late aborted group without poisoning the cache or new selection", async () => {
    let resolve!: (value: NormalizationStatistics) => void;
    const fetch = vi.fn((request: NormalizationGroup) =>
      request.group === 0
        ? new Promise<NormalizationStatistics>((done) => {
            resolve = done;
          })
        : Promise.resolve(result(request)),
    );
    const load = createNormalizationLoader(fetch);
    const controller = new AbortController();
    const stale = load(group, controller.signal);
    controller.abort();
    const current = await load({ ...group, group: 1 }, signal());
    expect(current.mean).toBe(1151.5);
    resolve(result());
    await expect(stale).rejects.toMatchObject({ name: "AbortError" });
    fetch.mockImplementation(async (request) => result(request));
    await load(group, signal());
    expect(fetch).toHaveBeenCalledTimes(3);
  });

  it("does not fetch or return cached statistics to an already aborted request", async () => {
    const fetch = vi.fn(async () => result());
    const load = createNormalizationLoader(fetch);
    await load(group, signal());
    const controller = new AbortController();
    controller.abort();
    await expect(load(group, controller.signal)).rejects.toMatchObject({
      name: "AbortError",
    });
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("retries failed requests and refuses mismatched or partial groups", async () => {
    const fetch = vi.fn(async () => result());
    const load = createNormalizationLoader(fetch);
    fetch.mockRejectedValueOnce(new Error("Snapshot unavailable"));
    await expect(load(group, signal())).rejects.toThrow("Snapshot unavailable");
    for (const changed of [
      { operation_id: "wrong" },
      { tensor_id: "wrong" },
      { group: 1 },
      { start: 8 },
      { count: 8 },
      { mean: null },
      { variance: -1 },
      { denominator: Infinity },
    ]) {
      fetch.mockResolvedValueOnce({ ...result(), ...changed });
      await expect(load(group, signal())).rejects.toThrow(
        "complete normalization group",
      );
    }
    expect((await load(group, signal())).mean).toBe(383.5);
    expect(fetch).toHaveBeenCalledTimes(10);
  });

  it("retains explicit non-finite and overflow states without fabricating statistics", async () => {
    for (const status of ["non_finite", "overflow"] as const) {
      const data = {
        ...result(),
        status,
        mean: null,
        variance: null,
        denominator: null,
      };
      const load = createNormalizationLoader(async () => data);
      expect(normalizationNumbers(await load(group, signal()))).toBeNull();
      expect(layerNormCalculation(null, 767, 1, 0)).toBeNull();
    }
    expect(normalizationNumbers({ ...result(), mean: NaN })).toBeNull();
    const zero = normalizationNumbers({
      ...result(),
      mean: 7,
      variance: 0,
      denominator: 0,
    });
    expect(zero).not.toBeNull();
    expect(layerNormCalculation(zero, 7, 1, 0)).toBeNull();
  });
});

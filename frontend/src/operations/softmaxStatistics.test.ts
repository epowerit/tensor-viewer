import { describe, expect, it, vi } from "vitest";
import type { SoftmaxStatistics } from "../api/client";
import {
  createSoftmaxLoader,
  validSoftmaxStatistics,
  type SoftmaxGroup,
} from "./softmaxStatistics";
const group: SoftmaxGroup = {
  runId: "r",
  operationId: "op",
  tensorId: "x",
  group: 0,
  size: 768,
};
const result = (p = group): SoftmaxStatistics => ({
  run_id: p.runId,
  operation_id: p.operationId,
  tensor_id: p.tensorId,
  group: p.group,
  count: p.size,
  status: "ok",
  maximum: 10 + p.group,
  denominator: 17.5,
  masked_count: 2,
});
const signal = () => new AbortController().signal;
describe("softmax full-group summaries", () => {
  it("caches by immutable group across score windows and bounds cache size", async () => {
    const fetch = vi.fn(async (p: SoftmaxGroup) => result(p));
    const load = createSoftmaxLoader(fetch, 2);
    for (const index of [0, 1, 0, 2, 0, 1])
      await load({ ...group, group: index }, signal());
    expect(fetch).toHaveBeenCalledTimes(4);
    for (const change of [
      { runId: "r2" },
      { operationId: "other" },
      { tensorId: "other" },
      { size: 1024 },
    ])
      await load({ ...group, ...change }, signal());
    expect(fetch).toHaveBeenCalledTimes(8);
  });
  it("rejects partial groups, stale identities, impossible denominators, and fabricated non-finite numbers", () => {
    for (const change of [
      { run_id: "bad" },
      { operation_id: "bad" },
      { tensor_id: "bad" },
      { group: 1 },
      { count: 8 },
      { maximum: null },
      { maximum: Infinity },
      { denominator: 0 },
      { denominator: 767 },
      { denominator: NaN },
      { masked_count: -1 },
      { masked_count: 768 },
      { masked_count: 1.5 },
      { status: "unknown" },
      { status: "all_masked" },
      { status: "non_finite" },
    ])
      expect(
        validSoftmaxStatistics(
          { ...result(), ...change } as SoftmaxStatistics,
          group,
        ),
      ).toBe(false);
    expect(
      validSoftmaxStatistics(
        {
          ...result(),
          status: "non_finite",
          maximum: null,
          denominator: null,
          masked_count: null,
        },
        group,
      ),
    ).toBe(true);
    expect(
      validSoftmaxStatistics(
        {
          ...result(),
          status: "all_masked",
          maximum: null,
          denominator: null,
          masked_count: 768,
        },
        group,
      ),
    ).toBe(true);
  });
  it("retries network and validation errors without caching failures", async () => {
    const fetch = vi
      .fn(async () => result())
      .mockRejectedValueOnce(new Error("Missing snapshot"))
      .mockResolvedValueOnce({ ...result(), count: 8 });
    const load = createSoftmaxLoader(fetch);
    await expect(load(group, signal())).rejects.toThrow("Missing snapshot");
    await expect(load(group, signal())).rejects.toThrow(
      "complete softmax group",
    );
    expect((await load(group, signal())).count).toBe(768);
  });
  it("discards late cancelled responses and never exposes a previous group's denominator", async () => {
    let resolve!: (v: SoftmaxStatistics) => void;
    const fetch = vi.fn((p: SoftmaxGroup) =>
      p.group === 0
        ? new Promise<SoftmaxStatistics>((done) => {
            resolve = done;
          })
        : Promise.resolve(result(p)),
    );
    const load = createSoftmaxLoader(fetch);
    const controller = new AbortController();
    const pending = load(group, controller.signal);
    controller.abort();
    expect((await load({ ...group, group: 1 }, signal())).maximum).toBe(11);
    resolve(result());
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    fetch.mockImplementation(async (p) => result(p));
    await load(group, signal());
    expect(fetch).toHaveBeenCalledTimes(3);
    await expect(load(group, controller.signal)).rejects.toMatchObject({
      name: "AbortError",
    });
  });
});

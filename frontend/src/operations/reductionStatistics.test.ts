import { describe, expect, it, vi } from "vitest";
import type { ReductionStatistics } from "../api/client";
import {
  createReductionLoader,
  reductionNumbers,
  validReductionStatistics,
  type ReductionGroup,
} from "./reductionStatistics";

const group: ReductionGroup = {
  runId: "run-1",
  operationId: "op-1",
  tensorId: "tensor-1",
  outputIndex: 0,
  size: 768,
  kind: "mean",
  integer: false,
};
const result = (request = group): ReductionStatistics => ({
  run_id: request.runId,
  operation_id: request.operationId,
  tensor_id: request.tensorId,
  output_index: request.outputIndex,
  count: request.size,
  status: "ok",
  sum: (383.5 + request.outputIndex * 768) * request.size,
  result: 383.5 + request.outputIndex * 768,
});
const signal = () => new AbortController().signal;

describe("complete reduction summaries", () => {
  it("uses every contributor rather than the visible window", async () => {
    const load = createReductionLoader(async (request) => result(request));
    const numbers = reductionNumbers(await load(group, signal()))!;
    expect(numbers).toEqual({ sum: 294528, result: 383.5 });
    expect(numbers.result).not.toBe((760 + 767) / 2);
  });

  it("reuses a group across contributor pages while separating runs, operations, outputs, and tensors", async () => {
    const fetch = vi.fn(async (request: ReductionGroup) => result(request));
    const load = createReductionLoader(fetch);
    for (let page = 0; page < 10; page++) await load(group, signal());
    expect(fetch).toHaveBeenCalledTimes(1);
    for (const change of [
      { outputIndex: 1 },
      { runId: "run-2" },
      { operationId: "op-2" },
      { tensorId: "tensor-2" },
      { size: 1536 },
    ])
      await load({ ...group, ...change }, signal());
    expect(fetch).toHaveBeenCalledTimes(6);
  });

  it("bounds its cache with least-recently-used eviction", async () => {
    const fetch = vi.fn(async (request: ReductionGroup) => result(request));
    const load = createReductionLoader(fetch, 2);
    for (const outputIndex of [0, 1, 0, 2, 0])
      await load({ ...group, outputIndex }, signal());
    expect(fetch).toHaveBeenCalledTimes(3);
    await load({ ...group, outputIndex: 1 }, signal());
    expect(fetch).toHaveBeenCalledTimes(4);
  });

  it("does not accept or cache late results from an abandoned output", async () => {
    let resolve!: (value: ReductionStatistics) => void;
    const fetch = vi.fn((request: ReductionGroup) =>
      request.outputIndex === 0
        ? new Promise<ReductionStatistics>((done) => {
            resolve = done;
          })
        : Promise.resolve(result(request)),
    );
    const load = createReductionLoader(fetch);
    const controller = new AbortController();
    const stale = load(group, controller.signal);
    controller.abort();
    expect((await load({ ...group, outputIndex: 1 }, signal())).result).toBe(
      1151.5,
    );
    resolve(result());
    await expect(stale).rejects.toMatchObject({ name: "AbortError" });
    fetch.mockImplementation(async (request) => result(request));
    await load(group, signal());
    expect(fetch).toHaveBeenCalledTimes(3);
    await expect(load(group, controller.signal)).rejects.toMatchObject({
      name: "AbortError",
    });
    expect(fetch).toHaveBeenCalledTimes(3);
  });

  it("retries failures and rejects partial groups, wrong identities, and invalid numbers", async () => {
    const fetch = vi.fn(async () => result());
    const load = createReductionLoader(fetch);
    fetch.mockRejectedValueOnce(new Error("Snapshot unavailable"));
    await expect(load(group, signal())).rejects.toThrow("Snapshot unavailable");
    for (const changes of [
      { run_id: "wrong" },
      { operation_id: "wrong" },
      { tensor_id: "wrong" },
      { output_index: 1 },
      { count: 8 },
      { result: null },
      { result: NaN },
      { sum: Infinity },
      { result: "nan" },
      { result: "123" },
      { status: "non_finite" },
      { status: "unknown" },
    ]) {
      fetch.mockResolvedValueOnce({
        ...result(),
        ...changes,
      } as ReductionStatistics);
      await expect(load(group, signal())).rejects.toThrow(
        "complete reduction group",
      );
    }
    expect((await load(group, signal())).result).toBe(383.5);
  });

  it("preserves large exact integer totals and rejects rounded integer JSON numbers", async () => {
    const integers: ReductionGroup = { ...group, kind: "sum", integer: true };
    const total = String(9007199254740993n * 768n);
    const data = { ...result(), sum: total, result: total };
    const load = createReductionLoader(async () => data);
    expect(reductionNumbers(await load(integers, signal()))).toEqual({
      sum: total,
      result: total,
    });
    for (const value of [
      9007199254740992,
      1.5,
      "1e20",
      "1.5",
      "nan",
      "9".repeat(40),
    ])
      expect(
        validReductionStatistics(
          { ...data, sum: value, result: value },
          integers,
        ),
      ).toBe(false);
    expect(
      validReductionStatistics(
        {
          ...data,
          sum: "-7083549724304467820544",
          result: "-7083549724304467820544",
        },
        integers,
      ),
    ).toBe(true);
    expect(
      validReductionStatistics({ ...data, sum: total, result: "1" }, integers),
    ).toBe(false);
  });

  it("distinguishes a finite mean with overflowing sum from a failed reference", async () => {
    const data: ReductionStatistics = { ...result(), sum: null, result: 1e308 };
    const load = createReductionLoader(async () => data);
    expect(reductionNumbers(await load(group, signal()))).toEqual({
      sum: undefined,
      result: 1e308,
    });
    expect(validReductionStatistics(data, { ...group, kind: "sum" })).toBe(
      false,
    );
    for (const status of ["non_finite", "overflow"] as const) {
      const failed = { ...result(), status, sum: null, result: null };
      expect(validReductionStatistics(failed, group)).toBe(true);
      expect(reductionNumbers(failed)).toBeNull();
      expect(validReductionStatistics({ ...failed, result: 0 }, group)).toBe(
        false,
      );
    }
  });
});

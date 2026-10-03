import { expect, test } from "vitest";
import type { Run } from "../api/client";
import {
  cellDeltas,
  closeness,
  matchingState,
  stateChange,
  summaryChange,
} from "./diff";

const state = (
  id: string,
  values: (number | string)[],
  shape = [values.length],
) => ({
  id,
  name: id,
  shape,
  dtype: "float32",
  values,
  numel: values.length,
});
const trace = (
  kinds: string[],
  tensors: ReturnType<typeof state>[],
): Run["trace"] =>
  ({
    input_ids: [tensors[0].id],
    operations: kinds.map((kind, index) => ({
      id: `op${index}`,
      index,
      kind,
      outputs: [tensors[index + 1].id],
    })),
    tensors: Object.fromEntries(tensors.map((tensor) => [tensor.id, tensor])),
  }) as unknown as Run["trace"];

test("a state lines up with the same input, or the same step's output", () => {
  const now = trace(
    ["relu", "sum"],
    [state("x", [1]), state("h", [2]), state("s", [3])],
  );
  const before = trace(
    ["relu", "mean"],
    [state("a", [0]), state("b", [1]), state("c", [9])],
  );
  expect(matchingState(now, before, "x")?.id).toBe("a");
  expect(matchingState(now, before, "h")?.id).toBe("b");
  // A different kind of step, or a different shape, does not line up.
  expect(matchingState(now, before, "s")).toBeNull();
  const reshaped = trace(
    ["relu", "sum"],
    [state("a", [0]), state("b", [1, 2]), state("c", [9])],
  );
  expect(matchingState(now, reshaped, "h")).toBeNull();
});

test("cell deltas count changes, including NaN appearing or going away", () => {
  const result = cellDeltas(
    state("now", [1.5, 2, "nan", 4, "nan"]),
    state("before", [1, 2, 3, "-inf", "nan"]),
  )!;
  expect(result.deltas).toEqual([0.5, 0, null, null, null]);
  expect([result.low, result.high]).toEqual([0, 0.5]);
  expect([result.changed, result.compared]).toEqual([3, 5]);
  // Paged values cannot be compared cell by cell.
  expect(
    cellDeltas({ values: [], numel: 4 }, state("b", [1, 2, 3, 4])),
  ).toBeNull();
});

test("a state's change since the run before reads as a summary", () => {
  const now = trace(
    ["relu", "sum"],
    [state("x", [1, 2]), state("h", [2, 2]), state("s", [3])],
  );
  const before = trace(
    ["relu", "mean"],
    [state("a", [1, 2]), state("b", [2, 5]), state("c", [9])],
  );
  expect(stateChange(now, before, "x")).toEqual({ kind: "same" });
  expect(stateChange(now, before, "h")).toEqual({
    kind: "changed",
    changed: 1,
    compared: 2,
  });
  expect(stateChange(now, before, "s")).toEqual({ kind: "new" });
});

test("snapshot tensors name their earlier state, for the backend to compare", () => {
  const paged = (id: string) => ({ ...state(id, []), shape: [4], numel: 4 });
  const now = trace(["relu"], [paged("x"), paged("h")]);
  const before = trace(["relu"], [paged("a"), paged("b")]);
  expect(stateChange(now, before, "h")).toEqual({
    kind: "unknown",
    before: "b",
  });
  expect(summaryChange({ changed: 3, compared: 4, low: 0, high: 1 })).toEqual({
    kind: "changed",
    changed: 3,
    compared: 4,
  });
  expect(summaryChange({ changed: 0, compared: 4, low: 0, high: 0 })).toEqual({
    kind: "same",
  });
  expect(summaryChange(null)).toBeNull();
});

test("closeness follows torch.allclose's defaults", () => {
  const summary = closeness(
    state("a", [1, 2, "inf"]),
    state("b", [1 + 1e-9, 2.5, "inf"]),
  )!;
  expect(summary.allclose).toBe(false);
  expect(summary.max_abs).toBe(0.5);
  expect(summary.mean_abs).toBeCloseTo((1e-9 + 0.5) / 2);
  expect(
    closeness(state("a", [1, "inf"]), state("b", [1 + 1e-9, "inf"]))?.allclose,
  ).toBe(true);
  // NaN is never close, even to NaN.
  expect(closeness(state("a", ["nan"]), state("b", ["nan"]))?.allclose).toBe(
    false,
  );
});

import { expect, test } from "vitest";
import type { Run } from "../api/client";
import {
  compareRuns,
  settleWithBackend,
  snapshotPairs,
  summarize,
} from "./compare";

type Step = [
  kind: string,
  shape: number[] | null,
  values?: (number | string)[],
];
function run(steps: Step[], failed = -1): Run {
  const tensors: Record<string, object> = {};
  const operations = steps.map(([kind, shape, values], i) => {
    if (shape)
      tensors[`t${i}`] = {
        name: `v${i}`,
        shape,
        dtype: "float32",
        numel: shape.reduce((a, b) => a * b, 1),
        values: values ?? [],
      };
    return {
      id: `op${i}`,
      kind,
      outputs: shape ? [`t${i}`] : [],
      status: i === failed ? "error" : "ok",
      source: { line: i + 1, text: kind },
    };
  });
  return { trace: { operations, tensors } } as unknown as Run;
}

test("identical runs match and value changes report the largest difference", () => {
  const base = run([
    ["reshape", [2, 2], [1, 2, 3, 4]],
    ["mul", [2, 2], [2, 4, 6, 8]],
  ]);
  const same = compareRuns(base, base);
  expect(same.map((s) => s.change)).toEqual(["same", "same"]);
  expect(summarize(same)).toContain("All 2 steps match");
  const scaled = run([
    ["reshape", [2, 2], [1, 2, 3, 4]],
    ["mul", [2, 2], [3, 6, 9, 12.5]],
  ]);
  const diff = compareRuns(base, scaled);
  expect(diff[1]).toMatchObject({ change: "values", delta: 4.5 });
  expect(summarize(diff)).toBe("1 of 2 steps differ, starting at step 2.");
});

test("shape, operation, status, and length differences are classified", () => {
  const left = run(
    [
      ["reshape", [2, 6]],
      ["permute", [6, 2]],
      ["matmul", null],
    ],
    2,
  );
  const right = run([
    ["reshape", [3, 4]],
    ["transpose", [6, 2]],
    ["matmul", [3, 3]],
    ["sum", [3]],
  ]);
  const diff = compareRuns(left, right);
  expect(diff.map((s) => s.change)).toEqual([
    "shape",
    "operation",
    "status",
    "added",
  ]);
  expect(compareRuns(right, left)[3].change).toBe("removed");
  expect(diff[3].left).toBeNull();
  expect(diff[2].left).toMatchObject({ failed: true, shape: null });
});

test("paged or non-finite values never produce an invented difference", () => {
  const paged = run([["reshape", [2, 2]]]);
  const inline = run([["reshape", [2, 2], [1, 2, 3, 4]]]);
  expect(compareRuns(paged, inline)[0]).toMatchObject({
    change: "same",
    delta: null,
    valuesCompared: false,
  });
  const nan = run([["div", [2], [1, "nan"]]]);
  const finite = run([["div", [2], [1, 2]]]);
  expect(compareRuns(nan, nan)[0].change).toBe("same");
  expect(compareRuns(nan, finite)[0]).toMatchObject({
    change: "values",
    delta: null,
    valuesCompared: true,
  });
  expect(summarize(compareRuns(run([]), run([])))).toContain("Neither run");
});

function withSecondOutput(
  base: Run,
  shape: number[],
  values: (number | string)[],
  dtype = "float32",
): Run {
  base.trace.tensors.second = {
    ...base.trace.tensors.t0,
    id: "second",
    name: "second",
    shape,
    dtype,
    numel: shape.reduce((a, b) => a * b, 1),
    values,
  };
  base.trace.operations[0].outputs.push("second");
  return base;
}

test("shape-only and paged traces report unavailable evidence in their summary", () => {
  const paged = run([["reshape", [2, 2]]]);
  const inline = run([["reshape", [2, 2], [1, 2, 3, 4]]]);
  const summary = summarize(compareRuns(paged, inline));
  expect(summary).toContain("Values were not fully compared for 1 step.");
  expect(summary).not.toContain("recorded values");
  // A zero-element shape-only tensor still carries no numeric capture evidence.
  const shapeOnly = run([["reshape", [0], []]]);
  shapeOnly.trace.tensors.t0.value_source = "shape";
  expect(compareRuns(shapeOnly, shapeOnly)[0].valuesCompared).toBe(false);
  shapeOnly.trace.tensors.t0.value_source = "paged";
  expect(compareRuns(shapeOnly, shapeOnly)[0].valuesCompared).toBe(false);
  const empty = run([["reshape", [0], []]]);
  expect(compareRuns(empty, empty)[0]).toMatchObject({
    change: "same",
    delta: 0,
    valuesCompared: true,
  });
});

test("every output participates in shape, type, count, and value comparisons", () => {
  const base = () =>
    withSecondOutput(run([["split", [2], [1, 2]]]), [2], [3, 4]);
  const values = withSecondOutput(run([["split", [2], [1, 2]]]), [2], [3, 10]);
  expect(compareRuns(base(), values)[0]).toMatchObject({
    change: "values",
    delta: 6,
    valuesCompared: true,
  });
  const shape = withSecondOutput(run([["split", [2], [1, 2]]]), [1, 2], [3, 4]);
  expect(compareRuns(base(), shape)[0].change).toBe("shape");
  const dtype = withSecondOutput(
    run([["split", [2], [1, 2]]]),
    [2],
    [3, 4],
    "int64",
  );
  expect(compareRuns(base(), dtype)[0].change).toBe("shape");
  expect(compareRuns(base(), run([["split", [2], [1, 2]]]))[0].change).toBe(
    "outputs",
  );
  expect(compareRuns(base(), base())[0].valuesCompared).toBe(true);
});

test("a known difference is retained when another output cannot be compared", () => {
  const left = withSecondOutput(run([["split", [2], []]]), [2], [3, 4]);
  const right = withSecondOutput(run([["split", [2], [1, 2]]]), [2], [3, 10]);
  const compared = compareRuns(left, right);
  expect(compared[0]).toMatchObject({
    change: "values",
    delta: null,
    valuesCompared: false,
  });
  expect(summarize(compared)).toBe(
    "1 of 1 steps differ, starting at step 1. Values were not fully compared for 1 step.",
  );
});

test("large integer strings differ exactly without pretending to measure their delta", () => {
  const left = run([["add", [1], ["9007199254740992"]]]);
  const right = run([["add", [1], ["9007199254740993"]]]);
  expect(compareRuns(left, right)[0]).toMatchObject({
    change: "values",
    delta: null,
    valuesCompared: true,
  });
  expect(compareRuns(left, left)[0]).toMatchObject({
    change: "same",
    delta: 0,
    valuesCompared: true,
  });
});

test("no-output operations do not claim a numeric value comparison", () => {
  const empty = run([["side_effect", null]]);
  const result = compareRuns(empty, empty);
  expect(result[0].valuesCompared).toBeNull();
  expect(summarize(result)).not.toContain("recorded values");
});

test("different distributions of large tensors prove a change; equal ones prove nothing", () => {
  const paged = (mean: number) => {
    const base = run([["linear", [64, 64]]]);
    Object.assign(base.trace.tensors.t0, {
      value_source: "paged",
      histogram: {
        low: -1,
        high: 1,
        counts: [10, 20],
        zeros: 0,
        non_finite: 0,
        mean,
        std: 0.5,
      },
    });
    return base;
  };
  expect(compareRuns(paged(0.1), paged(0.4))[0]).toMatchObject({
    change: "values",
    valuesCompared: false,
    shift: { before: 0.1, after: 0.4 },
  });
  expect(compareRuns(paged(0.1), paged(0.1))[0]).toMatchObject({
    change: "same",
    valuesCompared: false,
    shift: null,
  });
});

test("large tensors are settled by the backend's answers", () => {
  // Paged tensors record no values in the trace: the comparison is open.
  const paged = (id: string) => {
    const r = run([
      ["sub", [2, 2]],
      ["log", [2, 2], [1, 2, 3, 4]],
    ]);
    const t0 = (
      r.trace.tensors as Record<string, { value_source?: string; id?: string }>
    ).t0;
    t0.value_source = "paged";
    t0.id = id;
    (r.trace.tensors as Record<string, { id?: string }>).t1.id = `${id}-log`;
    return r;
  };
  const steps = compareRuns(paged("now"), paged("then"));
  expect(steps[0].valuesCompared).toBe(false);
  expect(summarize(steps)).toContain("not fully compared for 1 step");
  expect(snapshotPairs(steps)).toEqual([["now", "then"]]);
  const changed = settleWithBackend(
    steps,
    new Map([
      ["now|then", { changed: 3, compared: 4, low: -1, high: 2, max_abs: 2 }],
    ]),
  );
  expect(changed[0]).toMatchObject({
    change: "values",
    valuesCompared: true,
    delta: 2,
  });
  expect(summarize(changed)).not.toContain("not fully compared");
  const same = settleWithBackend(
    steps,
    new Map([
      ["now|then", { changed: 0, compared: 4, low: 0, high: 0, max_abs: 0 }],
    ]),
  );
  expect(same[0]).toMatchObject({ change: "same", valuesCompared: true });
  // Without an answer the step stays open.
  expect(settleWithBackend(steps, new Map())[0].valuesCompared).toBe(false);
});

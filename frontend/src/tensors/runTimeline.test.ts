import { expect, test } from "vitest";
import type { Run, Tensor } from "../api/client";
import { tensorAcrossRuns } from "./runTimeline";

type Trace = Run["trace"];

/** A run of steps: each a line of code, its result's width, and its kind. */
function run(id: string, steps: [string, number, string?][]) {
  const tensors: Record<string, Tensor> = {
    x: { id: "x", name: "x", shape: [2, 4] } as Tensor,
  };
  const operations = steps.map(([text, width, kind = "linear"], i) => {
    tensors[`${id}-t${i}`] = {
      id: `${id}-t${i}`,
      name: text,
      shape: [2, width],
      histogram: { std: width / 10 },
    } as unknown as Tensor;
    return {
      id: `op${i}`,
      kind,
      function: kind,
      module: "M",
      inputs: [],
      outputs: [`${id}-t${i}`],
      source: { line: i + 1, text, file: null },
    };
  });
  return {
    id,
    created_at: `2026-10-03T0${id.length}:00:00Z`,
    trace: { operations, tensors, input_ids: ["x"] } as unknown as Trace,
  };
}

test("a result is followed back through edits that add steps above it", () => {
  const now = run("now", [
    ["a = f(x)", 8],
    ["b = g(a)", 8],
    ["h = head(b)", 16],
  ]);
  const before = run("before", [
    ["b = g(a)", 8],
    ["h = head(b)", 12],
  ]);
  const first = run("first", [["h = head(b)", 12]]);
  const points = tensorAcrossRuns([now, before, first], "now-t2");
  expect(points.map((point) => point.tensor.id)).toEqual([
    "now-t2",
    "before-t1",
    "first-t0",
  ]);
  // The width changed in the last edit only.
  expect(points.map((point) => point.reshaped)).toEqual([true, false, false]);
});

test("the chain stops where the step did not exist; inputs match by name", () => {
  const now = run("now", [["a = f(x)", 8]]);
  const before = run("before", [["c = other(x)", 8, "relu"]]);
  expect(tensorAcrossRuns([now, before], "now-t0")).toHaveLength(1);
  expect(
    tensorAcrossRuns([now, before], "x").map((point) => point.runId),
  ).toEqual(["now", "before"]);
});

import { expect, test } from "vitest";
import type { Run } from "../api/client";
import {
  findValues,
  isBroken,
  landmarks,
  nextBroken,
  parseQuery,
  planeValues,
  previousOf,
  quantiles,
} from "./find";
import { tensorUses } from "./TensorUseContext";

const tensor = (values: (number | string)[], dtype = "float32") => ({
  values,
  numel: values.length,
  dtype,
  value_source: "inline" as const,
});

test("NaN and infinity are recognised as numbers or as Python's words", () => {
  expect(
    [NaN, -Infinity, "nan", "-inf", "Infinity", 0, "1.5"].map(isBroken),
  ).toEqual([true, true, true, true, true, false, false]);
});

test("landmarks find the first smallest, largest, and every non-finite value", () => {
  expect(landmarks(tensor([3, "nan", -2, 7, -2, 7, "-inf"]))).toEqual({
    min: 2,
    max: 3,
    broken: [1, 6],
  });
  expect(landmarks(tensor(["nan"]))).toEqual({
    min: null,
    max: null,
    broken: [0],
  });
  // Paged, shape-only, and boolean tensors have no landmarks to find.
  expect(landmarks({ ...tensor([1, 2]), numel: 4 })).toBeNull();
  expect(landmarks({ ...tensor([1]), value_source: "shape" })).toBeNull();
  expect(landmarks(tensor([1, 0], "bool"))).toBeNull();
});

test("the next non-finite value wraps around to the first", () => {
  expect(nextBroken([1, 6], 0)).toBe(1);
  expect(nextBroken([1, 6], 1)).toBe(6);
  expect(nextBroken([1, 6], 6)).toBe(1);
  expect(nextBroken([], 3)).toBeNull();
});

test("the plane on screen becomes its own small tensor", () => {
  const values = Array.from({ length: 24 }, (_, i) => i);
  const cube = { shape: [2, 3, 4], values, numel: 24, dtype: "float32" };
  expect(planeValues(cube, { row: 1, column: 2 }, [1, 0, 0])).toEqual({
    shape: [3, 4],
    dtype: "float32",
    values: values.slice(12),
    numel: 12,
  });
  // Swapped axes read the plane transposed.
  expect(planeValues(cube, { row: 2, column: 0 }, [0, 2, 0])?.values).toEqual([
    8, 20, 9, 21, 10, 22, 11, 23,
  ]);
});

test("each state knows the step that made it and the steps that read it", () => {
  const uses = tensorUses({
    operations: [
      { id: "op0", index: 0, kind: "linear", inputs: ["x"], outputs: ["h"] },
      { id: "op1", index: 1, kind: "relu", inputs: ["h"], outputs: ["a"] },
      {
        id: "op2",
        index: 2,
        kind: "add_",
        inputs: ["h", "a"],
        outputs: ["h2"],
        mutations: [{ before: "h", after: "h2", kind: "write" }],
      },
    ],
  } as unknown as Run["trace"]);
  expect(uses("x")).toEqual({
    made: null,
    read: [{ id: "op0", step: 1, kind: "linear" }],
  });
  expect(uses("h").read.map((step) => step.step)).toEqual([2, 3]);
  expect(uses("h2").made?.kind).toBe("add_");
});

test("value searches read comparisons, magnitudes, and non-finite words", () => {
  const values = [0, -3, 0.5, "nan", 2, "-inf", -0.25];
  const find = (text: string) => {
    const query = parseQuery(text);
    return "test" in query
      ? findValues({ values, numel: values.length }, query)
      : query.error;
  };
  expect(find("> 0.4")).toEqual([2, 4]);
  expect(find("0")).toEqual([0]);
  expect(find("= 0")).toEqual([0]);
  expect(find("!= 0")).toEqual([1, 2, 4, 6]);
  expect(find("abs > 1")).toEqual([1, 4]);
  expect(find("|x| <= .25")).toEqual([0, 6]);
  expect(find("nan")).toEqual([3]);
  expect(find("inf")).toEqual([5]);
  expect(find("finite")).toEqual([0, 1, 2, 4, 6]);
  expect(find("1e-3 <")).toBe("Try > 0.5, == 0, abs > 2, nan, or inf.");
  expect(find("  ")).toBe("");
  const query = parseQuery("abs >= 2");
  expect("label" in query && query.label).toBe("|x| >= 2");
});

test("stepping backwards wraps to the last match", () => {
  expect(previousOf([1, 6], 6)).toBe(1);
  expect(previousOf([1, 6], 1)).toBe(6);
  expect(previousOf([], 1)).toBeNull();
});

test("each query has the same question for the backend's search", () => {
  const spec = (text: string) => {
    const query = parseQuery(text);
    return "spec" in query ? query.spec : null;
  };
  expect(spec("abs >= 2")).toEqual({ test: "ge", value: 2, magnitude: true });
  expect(spec("0")).toEqual({ test: "eq", value: 0, magnitude: false });
  expect(spec("nan")).toEqual({ test: "nan" });
  expect(spec("finite")).toEqual({ test: "finite" });
  expect(spec("what")).toBeNull();
});

test("percentiles interpolate linearly, as numpy.percentile does", () => {
  const values = Array.from({ length: 101 }, (_, i) => i);
  expect(quantiles(values)).toEqual([1, 50, 99]);
  const [p1, median, p99] = quantiles([4, "nan", 1, 3, 2])!;
  expect(median).toBe(2.5);
  expect(p1).toBeCloseTo(1.03);
  expect(p99).toBeCloseTo(3.97);
  expect(quantiles(["nan"])).toBeNull();
});

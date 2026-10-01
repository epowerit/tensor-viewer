import { expect, test } from "vitest";
import type { Run } from "../api/client";
import {
  axisLineage,
  describeAxis,
  isOwnLineage,
  regroup,
  type AxisStory,
} from "./axisLineage";
import recorded from "./fixtures/lineage-trace.json";

// Recorded by the backend from a console script on x [2, 3, 8] with axes
// batch, tokens, features; values are omitted.
const trace = recorded as unknown as Run["trace"];
const byName = (name: string) =>
  Object.values(trace.tensors).find((tensor) => tensor.name === name)!;
const story = (name: string) =>
  axisLineage(trace, byName(name).id).map(describeAxis);

test("a head split is followed through reshape, permute, and back", () => {
  expect(story("q")).toEqual([
    "x.batch",
    "x.tokens",
    "x.features (piece 1 of 2×4)",
    "x.features (piece 2 of 2×4)",
  ]);
  expect(story("heads")).toEqual([
    "x.batch",
    "x.features (piece 1 of 2×4)",
    "x.tokens",
    "x.features (piece 2 of 2×4)",
  ]);
  // Rows come from the left operand and columns from the right.
  expect(story("scores")).toEqual([
    "x.batch",
    "x.features (piece 1 of 2×4)",
    "x.tokens",
    "x.tokens",
  ]);
  // Reshaping the pieces back together restores the original axis.
  expect(story("merged")).toEqual(["x.batch", "x.tokens", "x.features"]);
});

test("reductions, projections, joins, einsum, and broadcasts keep meaning", () => {
  expect(story("total")).toEqual(["x.batch", "x.features"]);
  expect(story("proj")).toEqual(["x.batch", "x.tokens", "w axis 0"]);
  expect(story("both")).toEqual([
    "x.batch",
    "x.tokens",
    "2 parts joined by cat",
  ]);
  expect(story("e")).toEqual(["x.batch", "x.tokens"]);
  expect(story("img")).toEqual([
    "an axis of size 1",
    "x.batch",
    "x.tokens",
    "x.features",
  ]);
  expect(story("shifted")).toEqual(["x.batch", "x.tokens", "x.features"]);
});

test("an operation without an axis-level story stops the trace honestly", () => {
  expect(story("conv")).toEqual([
    "computed by conv1d",
    "computed by conv1d",
    "computed by conv1d",
  ]);
  const input = byName("x");
  expect(isOwnLineage(input, axisLineage(trace, input.id))).toBe(true);
  expect(isOwnLineage(byName("q"), axisLineage(trace, byName("q").id))).toBe(
    false,
  );
  expect(axisLineage(trace, "missing")).toEqual([]);
});

test("regrouping matches element counts and keeps size-one axes in order", () => {
  const axes = (labels: string[], sizes: number[]): AxisStory[] =>
    labels.map((label, i) => ({
      size: sizes[i],
      terms: [{ label, size: sizes[i] }],
      note: null,
    }));
  const merged = regroup([2, 3, 4], [6, 4], axes(["a", "b", "c"], [2, 3, 4]));
  expect(merged.map(describeAxis)).toEqual(["a × b", "c"]);
  const ones = regroup([1, 12], [1, 3, 4], axes(["batch", "f"], [1, 12]));
  expect(ones.map(describeAxis)).toEqual([
    "batch",
    "f (piece 1 of 3×4)",
    "f (piece 2 of 3×4)",
  ]);
  // Splitting a merged axis differently has no single-axis story.
  const crossed = regroup([2, 6], [3, 4], axes(["a", "b"], [2, 6]));
  expect(crossed.map(describeAxis)).toEqual([
    "regrouped by reshape",
    "regrouped by reshape",
  ]);
});

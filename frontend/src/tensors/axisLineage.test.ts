import { expect, test } from "vitest";
import type { Run } from "../api/client";
import {
  axisLineage,
  describeAxis,
  lineageOf,
  isOwnLineage,
  regroup,
  scrambledAxes,
  unrelatedAxes,
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

// x [2, 3, 8] with axes batch, tokens, features, split into 2 heads of 4.
const whole = (label: string, size: number): AxisStory => ({
  size,
  terms: [{ label, size }],
  note: null,
});
const piece = (index: number): AxisStory => ({
  size: [2, 4][index],
  terms: [
    {
      label: "x.features",
      size: [2, 4][index],
      part: { index, of: 2, sizes: [2, 4] },
    },
  ],
  note: null,
});
const batch = whole("x.batch", 2),
  tokens = whole("x.tokens", 3);

test("reshaping heads back without a permute scrambles the axes", () => {
  // [batch, heads, tokens, head_dim] reshaped straight to [batch, tokens, features]
  expect(
    scrambledAxes([2, 2, 3, 4], [2, 3, 8], [batch, piece(0), tokens, piece(1)]),
  ).toEqual([
    {
      axes: [1, 2],
      sources: ["x.features piece 1", "x.tokens", "x.features piece 2"],
      reason: "regrouped",
    },
  ]);
  // Flattening it keeps the pieces apart, with tokens between them.
  expect(
    scrambledAxes([2, 2, 3, 4], [2, 24], [batch, piece(0), tokens, piece(1)])[0]
      ?.reason,
  ).toBe("reordered");
  // Swapped pieces merge into an axis that is not the original one.
  expect(
    scrambledAxes(
      [2, 3, 4, 2],
      [2, 3, 8],
      [batch, tokens, piece(1), piece(0)],
    )[0]?.reason,
  ).toBe("reordered");
});

test("ordinary merges and head batching are not flagged", () => {
  // Permuted back first: the pieces rejoin into x.features.
  expect(
    scrambledAxes([2, 3, 2, 4], [2, 3, 8], [batch, tokens, piece(0), piece(1)]),
  ).toEqual([]);
  // batch × heads, the usual way to batch attention heads.
  expect(
    scrambledAxes([2, 2, 3, 4], [4, 3, 4], [batch, piece(0), tokens, piece(1)]),
  ).toEqual([]);
  // batch × tokens, flattening whole axes in order.
  expect(
    scrambledAxes([2, 3, 8], [6, 8], [batch, tokens, whole("x.features", 8)]),
  ).toEqual([]);
  // Unknown lineage is never judged.
  expect(
    scrambledAxes(
      [2, 3, 8],
      [3, 16],
      [batch, tokens, { size: 8, terms: [], note: "computed by conv2d" }],
    ),
  ).toEqual([]);
});

test("a product is judged only between two traced input axes", () => {
  const data = (story: AxisStory): AxisStory => ({
    ...story,
    terms: story.terms.map((term) => ({ ...term, role: "input" as const })),
  });
  // q @ k without a transpose: features summed against tokens.
  expect(unrelatedAxes(data(piece(1)), data(tokens))).toBe(true);
  // q @ k.transpose(-2, -1): the same piece on both sides.
  expect(unrelatedAxes(data(piece(1)), data(piece(1)))).toBe(false);
  // x @ W: the weight axis is made to match, so it is never judged.
  const weight = {
    ...whole("w axis 0", 4),
    terms: [{ label: "w axis 0", size: 4, role: "parameter" as const }],
  };
  expect(unrelatedAxes(data(piece(1)), weight)).toBe(false);
  // Unknown lineage is never judged either.
  expect(
    unrelatedAxes(data(piece(1)), {
      size: 4,
      terms: [],
      note: "computed by conv2d",
    }),
  ).toBe(false);
});

test("the recorded attention scores pair matching axes", () => {
  const lookup = lineageOf(trace);
  const op = trace.operations.find((item) =>
    item.outputs.includes(byName("scores").id),
  )!;
  const [left, right] = op.inputs.map((id) => trace.tensors[id]);
  expect(
    unrelatedAxes(
      lookup(left.id)[left.shape.length - 1],
      lookup(right.id)[right.shape.length - 2],
    ),
  ).toBe(false);
});

test("lineage passes through steps that dedicated lessons present", () => {
  // relu and mean are shown by the activation and reduction lessons, which
  // carry the cell rule alongside; the payloads are as the backend records them.
  const tensor = (id: string, shape: number[], role = "intermediate") => ({
    id,
    name: id,
    shape,
    numel: shape.reduce((a, b) => a * b, 1),
    role,
    axes:
      id === "x"
        ? ["batch", "tokens", "features"]
        : shape.map((_, i) => `axis ${i}`),
    values: [],
    storage_id: id,
  });
  const steps = {
    input_ids: ["x"],
    tensors: {
      x: tensor("x", [2, 4, 8], "input"),
      h: tensor("h", [2, 4, 8]),
      pooled: tensor("pooled", [2, 8]),
    },
    operations: [
      {
        id: "op0",
        kind: "relu",
        status: "ok",
        inputs: ["x"],
        outputs: ["h"],
        arguments: {},
        lesson: {
          interaction: "activation",
          relation: { rule: "elementwise", operand: 0, roles: ["input"] },
        },
      },
      {
        id: "op1",
        kind: "mean",
        status: "ok",
        inputs: ["h"],
        outputs: ["pooled"],
        arguments: { dim: 1 },
        lesson: {
          interaction: "reduction",
          relation: { rule: "reduce", operand: 0, axes: [1], keepdim: false },
        },
      },
    ],
  } as unknown as Run["trace"];
  expect(axisLineage(steps, "pooled").map(describeAxis)).toEqual([
    "x.batch",
    "x.features",
  ]);
});

test("an in-place write keeps the axes of the state it overwrote", () => {
  const tensor = (id: string, role = "intermediate") => ({
    id,
    name: "x2",
    shape: [2, 3],
    numel: 6,
    role,
    axes: id === "x" ? ["rows", "columns"] : ["axis 0", "axis 1"],
    values: [],
    storage_id: id === "x" ? "s0" : "s1",
  });
  const steps = {
    input_ids: ["x"],
    tensors: {
      x: { ...tensor("x", "input"), name: "x" },
      a: tensor("a"),
      b: tensor("b"),
    },
    operations: [
      {
        id: "op0",
        kind: "clone",
        status: "ok",
        inputs: ["x"],
        outputs: ["a"],
        arguments: {},
        lesson: { interaction: "mapping", mapping_rule: "identity" },
      },
      {
        id: "op1",
        kind: "add_",
        status: "ok",
        inputs: ["a"],
        // In-place ops return the tensor they modify.
        outputs: ["b"],
        arguments: {},
        mutations: [{ kind: "write", before: "a", after: "b" }],
        lesson: { interaction: "inspect" },
      },
    ],
  } as unknown as Run["trace"];
  expect(axisLineage(steps, "b").map(describeAxis)).toEqual([
    "x.rows",
    "x.columns",
  ]);
});

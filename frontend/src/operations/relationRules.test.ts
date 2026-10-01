import { expect, test } from "vitest";
import type { Operation, Tensor } from "../api/client";
import { unravel } from "../tensors/coordinates";
import { layoutMorph, relationMorph } from "./layoutMorph";
import { presenters } from "./presenters";
import {
  classLossSamples,
  einsumTerms,
  prefixGroup,
  relationSource,
  relationTargets,
  tensorRelation,
} from "./relations";

const tensor = (name: string, shape: number[], values: number[] = []) =>
  ({
    id: name,
    name,
    shape,
    values,
    dtype: "float32",
    numel: shape.reduce((a, b) => a * b, 1),
  }) as Tensor;
const operation = (kind: string, relation: object, args: object = {}) =>
  ({
    kind,
    status: "ok",
    arguments: args,
    lesson: { interaction: "relation", relation, mapping: null },
  }) as unknown as Operation;

test("a running total covers the cells up to its own position", () => {
  const x = tensor("x", [2, 3], [1, 2, 3, 4, 5, 6]);
  const y = tensor("y", [2, 3], [1, 3, 6, 4, 9, 15]);
  const op = operation("cumsum", { rule: "prefix", operand: 0, axis: 1 });
  const rule = tensorRelation(op, [x], y) as Extract<
    ReturnType<typeof tensorRelation>,
    { rule: "prefix" }
  >;
  expect(prefixGroup(rule, x, 5)).toEqual({ indices: [3, 4, 5], size: 3 });
  expect(prefixGroup(rule, x, 3)).toEqual({ indices: [3], size: 1 });
  expect(prefixGroup(rule, x, 5, 2)).toEqual({ indices: [4, 5], size: 3 });
  const shown = presenters.relation(op, [x], y, 5);
  expect(shown.expression).toBe("4 + 5 + 6 = 15");
  expect(shown.leftHighlights).toEqual([3, 4, 5]);
  expect(relationTargets(rule, x, y, 4, 0)).toEqual([4]);
  expect(relationMorph(op, rule, x, y)).toBeNull();
  expect(
    tensorRelation(
      operation("cumsum", { rule: "prefix", operand: 0, axis: 2 }),
      [x],
      y,
    ),
  ).toBeNull();
});

test("padding separates original cells from the border", () => {
  const x = tensor("x", [2, 2], [1, 2, 3, 4]);
  const y = tensor("y", [2, 5], [0, 1, 2, 0, 0, 0, 3, 4, 0, 0]);
  const op = operation("pad", { rule: "pad", operand: 0, before: [0, 1] });
  const rule = tensorRelation(op, [x], y)!;
  expect([0, 1, 2, 3, 6].map((t) => relationSource(rule, x, y, t))).toEqual([
    undefined,
    0,
    1,
    undefined,
    2,
  ]);
  expect(relationTargets(rule, x, y, 3, 0)).toEqual([7]);
  expect(presenters.relation(op, [x], y, 0).title).toBe("A border cell");
  expect(presenters.relation(op, [x], y, 7).expression).toBe(
    "x[1, 1] → [1, 2] = 4",
  );
  expect(
    tensorRelation(
      operation("pad", { rule: "pad", operand: 0, before: [0, 4] }),
      [x],
      y,
    ),
  ).toBeNull();
});

test("einsum terms range over the letters the output drops", () => {
  // ij,kj->ik : a [2, 3] against b [2, 3], summed over j.
  const a = tensor("a", [2, 3], [1, 2, 3, 4, 5, 6]);
  const b = tensor("b", [2, 3], [1, 0, 1, 2, 2, 2]);
  const out = tensor("out", [2, 2], [4, 12, 10, 30]);
  const op = operation("einsum", {
    rule: "einsum",
    operand: 0,
    inputs: ["ij", "kj"],
    output: "ik",
    sizes: { i: 2, j: 3, k: 2 },
  });
  const rule = tensorRelation(op, [a, b], out) as Extract<
    ReturnType<typeof tensorRelation>,
    { rule: "einsum" }
  >;
  const found = einsumTerms(rule, [a.shape, b.shape], out.shape, 3);
  expect(found.size).toBe(3);
  expect(found.terms).toEqual([
    [3, 3],
    [4, 4],
    [5, 5],
  ]);
  // Every output equals the sum of its listed products.
  for (let target = 0; target < 4; target++) {
    const { terms } = einsumTerms(rule, [a.shape, b.shape], out.shape, target);
    expect(
      terms.reduce(
        (sum, [p, q]) => sum + Number(a.values[p]) * Number(b.values[q]),
        0,
      ),
    ).toBe(out.values[target]);
  }
  const shown = presenters.relation(op, [a, b], out, 1);
  expect(shown.operandHighlights).toEqual([
    [0, 1, 2],
    [3, 4, 5],
  ]);
  expect(shown.expression).toBe("(1 × 2) + (2 × 2) + (3 × 2) = 12");
  // Selecting a[1, 2] moves the output to row i = 1, keeping column k.
  expect(unravel(relationTargets(rule, a, out, 5, 1)[0], out.shape)).toEqual([
    1, 1,
  ]);
  const wrong = operation("einsum", {
    rule: "einsum",
    operand: 0,
    inputs: ["ij", "kj"],
    output: "ik",
    sizes: { i: 2, j: 4, k: 2 },
  });
  expect(tensorRelation(wrong, [a, b], out)).toBeNull();
});

test("a trace and a repeated letter are single-operand einsums", () => {
  const m = tensor("m", [3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const total = tensor("t", [], [15]);
  const op = operation("einsum", {
    rule: "einsum",
    operand: 0,
    inputs: ["ii"],
    output: "",
    sizes: { i: 3 },
  });
  const rule = tensorRelation(op, [m], total) as Extract<
    ReturnType<typeof tensorRelation>,
    { rule: "einsum" }
  >;
  expect(einsumTerms(rule, [m.shape], [], 0).terms).toEqual([[0], [4], [8]]);
  expect(presenters.relation(op, [m], total, 0).expression).toBe(
    "(1) + (5) + (9) = 15",
  );
  // Off-diagonal cells are not read, even though there is only one output.
  expect(relationTargets(rule, m, total, 1, 0)).toEqual([]);
  expect(relationTargets(rule, m, total, 4, 0)).toEqual([0]);
  const diagonal = tensor("diagonal", [3], [1, 5, 9]);
  const keep = { ...rule, output: "i" };
  expect(relationTargets(keep, m, diagonal, 5, 0)).toEqual([]);
  expect(relationTargets(keep, m, diagonal, 8, 0)).toEqual([2]);
});

test("either einsum operand can select a consuming output", () => {
  const a = tensor("a", [2, 3]);
  const b = tensor("b", [4, 3]);
  const out = tensor("out", [2, 4]);
  const op = operation("einsum", {
    rule: "einsum",
    operand: 0,
    inputs: ["ij", "kj"],
    output: "ik",
    sizes: { i: 2, j: 3, k: 4 },
  });
  const rule = tensorRelation(op, [a, b], out)!;
  // a[1,2] fixes row i=1 while preserving the current column k=3.
  expect(relationTargets(rule, a, out, 5, 3)).toEqual([7]);
  // b[2,1] fixes column k=2 while preserving the current row i=1.
  expect(relationTargets({ ...rule, operand: 1 }, b, out, 7, 7)).toEqual([6]);
});

test("axis aliases animate like the operations they stand for", () => {
  const alias = (kind: string, order: number[]) =>
    ({
      kind,
      status: "ok",
      arguments: {},
      mutations: [],
      lesson: {
        interaction: "mapping",
        mapping: null,
        mapping_rule: "permutation",
        axis_order: order,
      },
    }) as unknown as Operation;
  for (const kind of ["swapaxes", "movedim", "mT", "T"])
    expect(
      layoutMorph(alias(kind, [1, 0]), tensor("a", [2, 3]), tensor("b", [3, 2]))
        ?.sources,
    ).toEqual([0, 3, 1, 4, 2, 5]);
  expect(
    layoutMorph(
      alias("sort", [1, 0]),
      tensor("a", [2, 3]),
      tensor("b", [3, 2]),
    ),
  ).toBeNull();
});

test("one-hot cells compare the index with their class position", () => {
  const labels = { ...tensor("labels", [3], [2, 0, 1]), dtype: "int64" };
  const out = tensor("targets", [3, 4], [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0]);
  const op = operation("one_hot", { rule: "one_hot", operand: 0 });
  const hit = presenters.relation(op, [labels], out, 2);
  expect(hit.leftHighlights).toEqual([0]);
  expect(hit.title).toBe("This is the named class: 1");
  expect(hit.expression).toBe("2 == 2 → 1");
  expect(presenters.relation(op, [labels], out, 5).expression).toBe(
    "0 == 1 → 0",
  );
  const rule = tensorRelation(op, [labels], out)!;
  // Selecting the third label keeps the class column on screen.
  expect(relationTargets(rule, labels, out, 2, 5)).toEqual([9]);
  expect(tensorRelation(op, [tensor("f", [3], [2, 0, 1])], out)).toBeNull();
});

test("a class loss shows each sample's target score and the reduction", () => {
  const scores = tensor("x", [2, 3], [0, 0, 0, 0, Math.log(2), 0]);
  const labels = { ...tensor("labels", [2], [0, 1]), dtype: "int64" };
  // Row 0 is uniform: −log(1/3). Row 1 has p(class 1) = 2/4: −log(1/2).
  const first = Math.log(3),
    second = Math.log(2);
  const make = (reduction: string, shape: number[], values: number[]) => ({
    op: operation("cross_entropy", {
      rule: "class_loss",
      operand: 0,
      reduction,
      ignore_index: -100,
      normalized: false,
    }),
    out: tensor("loss", shape, values),
  });
  const mean = make("mean", [], [(first + second) / 2]);
  const shown = presenters.relation(mean.op, [scores, labels], mean.out, 0);
  expect(shown.operandHighlights).toEqual([
    [0, 4],
    [0, 1],
  ]);
  expect(shown.title).toBe("The average over 2 samples");
  expect(shown.expression).toBe("mean(1.099, 0.693) = 0.896");
  const none = make("none", [2], [first, second]);
  const single = presenters.relation(none.op, [scores, labels], none.out, 1);
  expect(single.operandHighlights).toEqual([[4], [1]]);
  expect(single.expression).toBe("−log p(class 1) = 0.693");
  // An ignored target is skipped; a wrong output shape is rejected.
  const skipped = {
    ...labels,
    values: [-100, 1],
  };
  expect(
    presenters.relation(mean.op, [scores, skipped], mean.out, 0)
      .operandHighlights![0],
  ).toEqual([4]);
  expect(
    tensorRelation(mean.op, [scores, labels], tensor("loss", [2], [1, 1])),
  ).toBeNull();
  // Log-probabilities are used as they are.
  const nll = operation("nll_loss", {
    rule: "class_loss",
    operand: 0,
    reduction: "sum",
    ignore_index: -100,
    normalized: true,
  });
  const logp = tensor("lp", [2, 3], [-1, -2, -3, -4, -5, -6]);
  expect(
    presenters.relation(nll, [logp, labels], tensor("loss", [], [6]), 0)
      .expression,
  ).toBe("sum(1, 5) = 6");
});

test("large class-loss previews keep the whole batch distinct from shown samples", () => {
  const scores = tensor("scores", [1024, 1_000_000]);
  const labels = {
    ...tensor("labels", [1024], Array(1024).fill(0)),
    dtype: "int64",
  };
  const op = operation("cross_entropy", {
    rule: "class_loss",
    operand: 0,
    reduction: "mean",
    ignore_index: -100,
    normalized: false,
  });
  const out = tensor("loss", [], [13.816]);
  const shown = presenters.relation(op, [scores, labels], out, 0);
  expect(shown.title).toBe("The average over 1,024 samples");
  expect(shown.text).toContain("first 256 of 1,024 samples");
  expect(shown.text).toContain("recorded result includes the entire batch");
  expect(shown.expression).toBeUndefined();
  expect(shown.operandHighlights![1]).toHaveLength(256);
  // An ignored target beyond the preview must still affect the exact count.
  labels.values[1000] = -100;
  expect(presenters.relation(op, [scores, labels], out, 0).title).toBe(
    "The average over 1,023 samples",
  );
  const unknown = presenters.relation(
    op,
    [scores, { ...labels, values: [] }],
    out,
    0,
  );
  expect(unknown.title).toBe("The average over non-ignored targets");
  expect(unknown.text).toContain("number of contributing samples is unknown");
  expect(unknown.expression).toBeUndefined();
});

test("class-loss arithmetic remains bounded and never fabricates a complete equality", () => {
  const scores = tensor("scores", [256, 32], Array(8192).fill(0));
  const labels = {
    ...tensor("labels", [256], Array(256).fill(0)),
    dtype: "int64",
  };
  const op = operation("cross_entropy", {
    rule: "class_loss",
    operand: 0,
    reduction: "mean",
    ignore_index: -100,
    normalized: false,
  });
  const out = tensor("loss", [], [Math.log(32)]);
  const rule = tensorRelation(op, [scores, labels], out) as Extract<
    ReturnType<typeof tensorRelation>,
    { rule: "class_loss" }
  >;
  const samples = classLossSamples(rule, scores, labels, 0);
  expect(samples.filter((sample) => sample.loss !== null)).toHaveLength(128);
  expect(samples[0].loss).toBeCloseTo(Math.log(32));
  expect(
    presenters.relation(op, [scores, labels], out, 0).expression,
  ).toBeUndefined();
  const ignored = { ...labels, values: Array(256).fill(-100) };
  const noTargets = presenters.relation(
    op,
    [scores, ignored],
    { ...out, values: ["nan"] },
    0,
  );
  expect(noTargets.title).toBe("No contributing samples");
  expect(noTargets.expression).toBe("All targets ignored → nan");
});

test("broadcast alignment marks reused and missing axes from the right", async () => {
  const { alignShapes } = await import("./relations");
  const rows = alignShapes([[3, 1], [4]], [3, 4])!;
  expect(rows[0]).toEqual([
    { size: 3, stretched: false },
    { size: 1, stretched: true },
  ]);
  expect(rows[1]).toEqual([
    { size: null, stretched: true },
    { size: 4, stretched: false },
  ]);
  // Equal shapes, and a unit axis that stays a unit axis, need no explanation.
  expect(
    alignShapes(
      [
        [2, 3],
        [2, 3],
      ],
      [2, 3],
    ),
  ).toBeNull();
  expect(alignShapes([[1, 3]], [1, 3])).toBeNull();
  // expand: one operand against a larger output.
  expect(alignShapes([[1, 3]], [2, 2, 3])![0].map((a) => a.stretched)).toEqual([
    true,
    true,
    false,
  ]);
});

test("batch normalization in evaluation mode reads one statistic per channel", () => {
  const x = tensor("x", [2, 2], [5, 4, 9, 6]);
  const out = tensor("y", [2, 2], [2, 12, 4, 16]);
  const operands = [
    x,
    tensor("mean", [2], [1, 3]),
    tensor("var", [2], [4, 1]),
    tensor("scale", [2], [1, 2]),
    tensor("shift", [2], [0, 10]),
  ];
  const op = operation("batch_norm", {
    rule: "channel_affine",
    operand: 0,
    axis: 1,
    roles: ["input", "running_mean", "running_var", "weight", "bias"],
    eps: 0,
  });
  const shown = presenters.relation(op, operands, out, 3);
  expect(shown.operandHighlights).toEqual([[3], [1], [1], [1], [1]]);
  expect(shown.title).toBe("Channel 1's stored statistics");
  expect(shown.expression).toBe("(6 − 3) ÷ √(1 + 0) × 2 + 10 ≈ 16");
  const plain = operation("batch_norm", {
    rule: "channel_affine",
    operand: 0,
    axis: 1,
    roles: ["input", "running_mean", "running_var"],
    eps: 0,
  });
  expect(
    presenters.relation(
      plain,
      operands.slice(0, 3),
      tensor("y", [2, 2], [2, 1, 4, 3]),
      0,
    ).expression,
  ).toBe("(5 − 1) ÷ √(4 + 0) ≈ 2");
  // Statistics of the wrong length are rejected.
  expect(
    tensorRelation(
      plain,
      [x, tensor("mean", [3], [1, 2, 3]), operands[2]],
      out,
    ),
  ).toBeNull();
});

import { expect, test } from "vitest";
import type { Run } from "../api/client";
import { explainTensor } from "./TensorStory";

const tensor = (id: string, name: string, shape: number[], extra = {}) => ({
  id,
  name,
  shape,
  numel: shape.reduce((a, b) => a * b, 1),
  dtype: "float32",
  axes: shape.map((_, i) => `axis ${i}`),
  storage_id: `s-${id}`,
  role: "intermediate",
  value_source: "inline",
  values: [],
  ...extra,
});

test("a tensor explained: origin, axes, values, memory, readers, and notes", () => {
  const trace = {
    input_ids: ["x"],
    output_ids: ["w"],
    tensors: {
      x: tensor("x", "x", [2, 4], {
        role: "input",
        axes: ["batch", "features"],
      }),
      s: tensor("s", "scores", [2, 4], {
        axes: ["batch", "features"],
        minimum: -1,
        maximum: 3,
        histogram: {
          low: -1,
          high: 3,
          counts: [6, 2],
          zeros: 2,
          non_finite: 0,
          mean: 0.5,
          std: 1.2,
        },
      }),
      v: tensor("v", "flat", [8], { storage_id: "s-s" }),
      w: tensor("w", "weights", [8]),
    },
    operations: [
      {
        id: "op0",
        index: 0,
        kind: "mul",
        inputs: ["x"],
        outputs: ["s"],
        source: { line: 7 },
      },
      {
        id: "op1",
        index: 1,
        kind: "view",
        inputs: ["s"],
        outputs: ["v"],
        source: { line: 8 },
      },
      {
        id: "op2",
        index: 2,
        kind: "softmax",
        inputs: ["v"],
        outputs: ["w"],
        source: { line: 9 },
      },
    ],
  } as unknown as Run["trace"];
  const scores = explainTensor(trace, trace.tensors.s, {
    symbolic: ["B", "4"],
    notes: [{ node: "op0", title: "Something to see" } as never],
  });
  expect(scores).toEqual([
    "scores is made at step 1 by mul (line 7) from x [2, 4].",
    "Its axes are batch (2) and features (4).",
    "In the input's sizes it is [B, 4].",
    "Its values run from -1 to 3, mean 0.5, σ 1.2; 25% are zero.",
    "It takes 32 B of memory.",
    "It is read by step 2 (view → flat).",
    "Run notes on its step: Something to see.",
  ]);
  const flat = explainTensor(trace, trace.tensors.v);
  expect(flat).toContain(
    "It is a view of scores, so its 32 B are not new memory.",
  );
  expect(explainTensor(trace, trace.tensors.w).at(-1)).toBe(
    "It is one of the model's results.",
  );
  expect(explainTensor(trace, trace.tensors.x)[0]).toBe(
    "x is an input of the model, [2, 4] float32.",
  );
});

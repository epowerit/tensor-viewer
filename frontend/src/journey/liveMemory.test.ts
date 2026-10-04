import { expect, test } from "vitest";
import type { Run } from "../api/client";
import { liveMemory } from "./liveMemory";

/** float32 tensors of n elements: 4·n bytes each. */
const tensor = (id: string, numel: number, extra = {}) => ({
  id,
  name: id,
  numel,
  dtype: "float32",
  storage_id: id,
  role: "intermediate",
  ...extra,
});

test("each storage lives from the step that makes it to the last that reads it", () => {
  // x → a (reads x) → v (a view of a) → b (reads v) → out (reads b)
  const trace = {
    input_ids: ["x"],
    output_ids: ["out"],
    tensors: {
      x: tensor("x", 10, { role: "input" }),
      w: tensor("w", 1000, { role: "parameter" }),
      wt: tensor("wt", 1000, { storage_id: "w" }),
      a: tensor("a", 100),
      v: tensor("v", 100, { storage_id: "a" }),
      b: tensor("b", 20),
      out: tensor("out", 5),
    },
    operations: [
      { id: "op0", inputs: ["x", "w"], outputs: ["a"] },
      // A view of the weight (weight.T) is not an activation either.
      { id: "opT", inputs: ["w"], outputs: ["wt"] },
      { id: "op1", inputs: ["a"], outputs: ["v"] },
      { id: "op2", inputs: ["v"], outputs: ["b"] },
      { id: "op3", inputs: ["b"], outputs: ["out"] },
    ],
  } as unknown as Run["trace"];
  const { live, peak } = liveMemory(trace);
  // Weights are not activations; the view shares a's storage; the input is
  // held throughout; a is freed after op2 reads its view.
  expect([...live.values()]).toEqual([
    40 + 400,
    40 + 400,
    40 + 400,
    40 + 400 + 80,
    40 + 80 + 20,
  ]);
  expect(peak).toMatchObject({ opId: "op2", step: 4, bytes: 520 });
  expect(peak!.holders.map((h) => h.tensorId)).toEqual(["a", "b", "x"]);
});

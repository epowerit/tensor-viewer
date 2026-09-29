import { describe, it, expect } from "vitest";
import type { ModuleCall, Operation, Run, Tensor } from "../api/client";
import {
  windowScores,
  windowScoreSelection,
  selectScoreOperand,
} from "./windowScores";
function fixture() {
  const shapes: Record<string, number[]> = {
    scores: [8, 2, 4, 4],
    table: [9, 2],
    lookup: [4, 4],
    relative: [4, 4, 2],
    bias: [2, 4, 4],
    positioned: [8, 2, 4, 4],
    grouped: [2, 4, 2, 4, 4],
    mask: [4, 1, 4, 4],
    masked: [2, 4, 2, 4, 4],
    flat: [8, 2, 4, 4],
    weights: [8, 2, 4, 4],
  };
  const tensors: Record<string, Tensor> = Object.fromEntries(
    Object.entries(shapes).map(([id, shape]) => [
      id,
      {
        id,
        name: id,
        shape,
        dtype: id === "mask" ? "bool" : id === "lookup" ? "int64" : "float32",
        numel: shape.reduce((a, b) => a * b, 1),
        value_source: "shape",
        values: [],
        axes: [],
        strides: [],
        storage_id: id,
        storage_offset: 0,
        contiguous: true,
        minimum: null,
        maximum: null,
        role: "intermediate",
      } satisfies Tensor,
    ]),
  );
  const rows = [
    ["__getitem__", ["table", "lookup"], "relative"],
    ["permute", ["relative"], "bias"],
    ["add", ["scores", "bias"], "positioned"],
    ["reshape", ["positioned"], "grouped"],
    ["masked_fill", ["grouped", "mask"], "masked"],
    ["reshape", ["masked"], "flat"],
    ["softmax", ["flat"], "weights"],
  ] as const;
  const operations = rows.map(
    ([kind, inputs, out], index) =>
      ({
        id: `op${index}`,
        index,
        kind,
        inputs: [...inputs],
        outputs: [out],
        status: "ok",
        arguments:
          kind === "masked_fill"
            ? { value: "-inf" }
            : kind === "softmax"
              ? { dim: -1 }
              : {},
        lesson: {
          axis_order: kind === "permute" ? [2, 0, 1] : null,
          mapping_rule: kind === "reshape" ? "identity" : null,
        },
      }) as Operation,
  );
  const call: ModuleCall = {
    id: "call",
    parent_id: null,
    path: "stage_0.attention.adjust",
    module_type: "WindowScoreAdjustment",
    start_index: 0,
    end_index: 7,
    inputs: ["scores"],
    outputs: ["weights"],
  };
  const run = { trace: { tensors, operations, module_calls: [call] } } as Run;
  return { run, call, p: windowScores(run, call)! };
}
describe("recorded window score adjustment", () => {
  it("links query, key, head, mask window and bias lookup without losing batch", () => {
    const { p } = fixture();
    expect(p).not.toBeNull();
    const s = windowScoreSelection(p, 255);
    expect(s).toMatchObject({
      batch: 1,
      window: 3,
      head: 1,
      query: 3,
      key: 3,
      bias: 31,
      mask: 63,
      lookup: 15,
      queryPosition: [1, 1],
      keyPosition: [1, 1],
    });
    expect(selectScoreOperand(p, 2, "bias", 255)).toBe(226);
    expect(selectScoreOperand(p, 6, "mask", 255)).toBe(150);
  });
  it("requires the real lookup, bias, mask, reshape, softmax dependency chain", () => {
    const { run, call } = fixture();
    run.trace.operations[4].inputs[0] = "scores";
    expect(windowScores(run, call)).toBeNull();
  });
  it("rejects nonnegative masking, different softmax axes and incomplete calls", () => {
    for (const change of [
      (r: Run) => {
        r.trace.operations[4].arguments.value = 0;
      },
      (r: Run) => {
        r.trace.operations[6].arguments.dim = 1;
      },
      (r: Run) => {
        r.trace.operations.pop();
      },
    ]) {
      const { run, call } = fixture();
      change(run);
      expect(windowScores(run, call)).toBeNull();
    }
  });
  it("rejects a mutated lookup or incompatible buffer dtype", () => {
    const { run, call } = fixture();
    run.trace.tensors.mask.dtype = "float32";
    expect(windowScores(run, call)).toBeNull();
  });
});

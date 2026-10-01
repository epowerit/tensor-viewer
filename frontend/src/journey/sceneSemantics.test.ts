import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import { operationSemantics } from "./sceneSemantics";

const tensor = (id: string, shape: number[], role = "intermediate"): Tensor =>
  ({
    id,
    shape,
    dtype: "float32",
    numel: shape.reduce((a, b) => a * b, 1),
    role,
  }) as Tensor;
const lesson = (value: object) => value as Operation["lesson"];
function fixture(
  kind: string,
  inputs: Tensor[],
  outputs: Tensor[],
  extra: Partial<Operation> = {},
) {
  const op = {
    kind,
    inputs: inputs.map((t) => t.id),
    outputs: outputs.map((t) => t.id),
    status: "ok",
    arguments: {},
    lesson: {},
    ...extra,
  } as Operation;
  const run = {
    trace: {
      tensors: Object.fromEntries(
        [...inputs, ...outputs].map((t) => [t.id, t]),
      ),
    },
  } as Run;
  return { run, op, read: () => operationSemantics(run, op) };
}

describe("recorded operation scene semantics", () => {
  it("describes a huge reshape using metadata without inspecting cells", () => {
    const input = tensor("x", [1024, 16, 768]);
    Object.defineProperty(input, "values", {
      get() {
        throw Error("No cell scans");
      },
    });
    const f = fixture("reshape", [input], [tensor("y", [1024, 12288])], {
      lesson: lesson({ interaction: "mapping", mapping_rule: "identity" }),
    });
    expect(f.read().summary).toContain("12582912 elements");
    expect(f.read().summary).toContain("[1024 × 16 × 768] → [1024 × 12288]");
    expect(f.read().summary).not.toContain("batch");
  });

  it("uses a recorded axis order, not a guessed shape correspondence", () => {
    const f = fixture(
      "permute",
      [tensor("x", [2, 3, 2])],
      [tensor("y", [2, 2, 3])],
      {
        lesson: lesson({
          interaction: "mapping",
          mapping_rule: "permutation",
          axis_order: [2, 0, 1],
        }),
      },
    );
    expect(f.read().summary).toContain("[2, 0, 1]");
    f.op.lesson.axis_order = [0, 0, 1];
    expect(f.read().title).not.toBe("Reorder the axes");
    f.op.lesson.axis_order = [0, 1, 2];
    expect(f.read().title).not.toBe("Reorder the axes");
  });

  it("does not call a dtype view a value-preserving reshape", () => {
    const f = fixture(
      "view",
      [tensor("x", [4])],
      [{ ...tensor("y", [4]), dtype: "int32" }],
      {
        lesson: lesson({
          interaction: "mapping",
          mapping_rule: "identity",
          title: "Reinterpret the stored bits",
        }),
      },
    );
    expect(f.read().title).toBe("Reinterpret the stored bits");
    expect(f.read().summary).not.toContain("same logical order");
  });

  it.each([
    [[2, 3, 4], [1, 4, 5], [2, 3, 5], 4],
    [[4], [4], [], 4],
    [[4], [2, 4, 3], [2, 3], 4],
  ])(
    "validates batched and vector matrix products",
    (left, right, output, terms) => {
      const f = fixture(
        "matmul",
        [tensor("a", left as number[]), tensor("b", right as number[])],
        [tensor("y", output as number[])],
      );
      expect(f.read().inputs.map((i) => i.label)).toEqual([
        "Left operand",
        "Right operand",
      ]);
      expect(f.read().summary).toContain(`${terms} matching pairs`);
      f.run.trace.tensors.y.shape = [7, 7];
      expect(f.read().title).not.toBe("Multiply and accumulate");
    },
  );

  it("distinguishes learned operands from the activation input", () => {
    const f = fixture(
      "linear",
      [
        tensor("x", [2, 4, 8]),
        tensor("w", [16, 8], "parameter"),
        tensor("b", [16], "parameter"),
      ],
      [tensor("y", [2, 4, 16])],
    );
    expect(f.read().inputs.map((i) => i.label)).toEqual([
      "Input",
      "Weights",
      "Bias",
    ]);
    expect(f.read().summary).toContain(
      "8 input features become 16 output features",
    );
    expect(f.read().summary).toContain("with bias");
  });

  it("describes verified convolution kernels, strides, and groups", () => {
    const f = fixture(
      "conv2d",
      [tensor("image", [2, 4, 8, 8]), tensor("kernel", [6, 2, 3, 3])],
      [tensor("y", [2, 6, 3, 3])],
      { arguments: { stride: 2, groups: 2 } },
    );
    expect(f.read().inputs[1].label).toBe("Kernel");
    expect(f.read().summary).toContain("2 channel groups");
    expect(f.read().summary).toContain("18 input × weight terms");
  });

  it.each(["split", "chunk", "unbind"])(
    "keeps every %s result visible in the roles",
    (kind) => {
      const f = fixture(
        kind,
        [tensor("x", [2, 3])],
        [
          tensor("a", kind === "unbind" ? [3] : [1, 3]),
          tensor("b", kind === "unbind" ? [3] : [1, 3]),
        ],
        { arguments: { dim: 0 } },
      );
      expect(f.read().outputs).toEqual([
        { tensorId: "a", label: "Part 1" },
        { tensorId: "b", label: "Part 2" },
      ]);
      expect(f.read().summary).toContain("along axis 0");
      expect(f.read().summary.includes("axis is removed")).toBe(
        kind === "unbind",
      );
    },
  );

  it("preserves repeated concatenation operands and distinguishes stacking", () => {
    const x = tensor("x", [2, 3]);
    const cat = fixture("cat", [x, x], [tensor("y", [2, 6])], {
      arguments: { dim: -1 },
    });
    expect(cat.read().inputs).toEqual([
      { tensorId: "x", label: "Part 1" },
      { tensorId: "x", label: "Part 2" },
    ]);
    expect(cat.read().summary).toContain("along axis 1");
    const stack = fixture("stack", [x, x], [tensor("y", [2, 3, 2])], {
      arguments: { dim: -1 },
    });
    expect(stack.read().summary).toContain("new axis 2");
  });

  it("reports exactly which operand axes broadcast", () => {
    const f = fixture(
      "add",
      [tensor("x", [2, 4, 8]), tensor("bias", [8])],
      [tensor("y", [2, 4, 8])],
    );
    expect(f.read().summary).toBe(
      "Right operand reuses values along axes 0, 1. Result [2 × 4 × 8].",
    );
    f.run.trace.tensors.bias.shape = [7];
    expect(f.read().summary).not.toContain("reuses");
  });

  it("describes reduction axes and retained dimensions from a validated rule", () => {
    const f = fixture(
      "mean",
      [tensor("x", [2, 4, 8])],
      [tensor("y", [2, 1, 8])],
      {
        lesson: lesson({
          interaction: "relation",
          relation: { rule: "reduce", operand: 0, axes: [1], keepdim: true },
        }),
      },
    );
    expect(f.read().summary).toContain("mean combines values along axis 1");
    expect(f.read().summary).toContain("Reduced axes remain at size 1");
    f.op.lesson.relation = {
      rule: "reduce",
      operand: 0,
      axes: [2],
      keepdim: true,
    };
    expect(f.read().summary).not.toContain("combines values along");
  });

  it("does not invent meanings for custom tuple results or unknown operations", () => {
    const f = fixture(
      "custom_rnn",
      [tensor("x", [2, 4, 8])],
      [tensor("a", [2, 4, 8]), tensor("b", [2, 8])],
    );
    expect(f.read().outputs.map((o) => o.label)).toEqual([
      "Result 1",
      "Result 2",
    ]);
    expect(f.read().summary).toBe(
      "custom_rnn returns 2 recorded tensors. Each result has its own shape and values.",
    );
    const unknown = fixture(
      "fused_custom",
      [tensor("x", [2])],
      [tensor("y", [2])],
    );
    expect(unknown.read().summary).toBe(
      "1 recorded tensor input → 1 tensor result.",
    );
  });

  it("prioritizes failure over incomplete shape geometry", () => {
    const f = fixture("matmul", [tensor("x", [2, 3])], [], {
      status: "error",
      error: "Shape mismatch",
    });
    expect(f.read().title).toBe("Operation stopped");
    expect(f.read().summary).toContain("did not complete");
    expect(f.read().outputs).toEqual([]);
  });

  it("orders returned and side-effect tensors like the canvas projection", () => {
    const f = fixture("add_", [tensor("x", [2])], [tensor("y", [2])], {
      mutations: [
        { before: "x", after: "y", kind: "write" },
        { before: "view_before", after: "view_after", kind: "alias" },
        { before: "other_before", after: "other_after", kind: "write" },
      ],
    });
    expect(f.read().inputs.map((i) => i.tensorId)).toEqual([
      "x",
      "view_before",
      "other_before",
    ]);
    expect(f.read().outputs).toEqual([
      { tensorId: "y", label: "Updated tensor" },
      { tensorId: "other_after", label: "Updated tensor" },
      { tensorId: "view_after", label: "Shared view" },
    ]);
  });
});

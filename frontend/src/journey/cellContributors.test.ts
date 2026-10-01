import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import {
  CELL_CONTRIBUTOR_LIMIT,
  traceCellContributors,
} from "./cellContributors";

const tensor = (id: string, shape: number[], values: number[] = []): Tensor =>
  ({
    id,
    shape,
    values,
    numel: shape.reduce((a, b) => a * b, 1),
    dtype: "float32",
    strides: shape.map((_, axis) =>
      shape.slice(axis + 1).reduce((a, b) => a * b, 1),
    ),
    storage_offset: 0,
  }) as Tensor;
function fixture(
  kind: string,
  inputs: Tensor[],
  outputs: Tensor[],
  extra: Partial<Operation> = {},
) {
  const op = {
    kind,
    status: "ok",
    arguments: {},
    mutations: [],
    inputs: inputs.map((t) => t.id),
    outputs: outputs.map((t) => t.id),
    lesson: { interaction: "inspect" },
    ...extra,
  } as Operation;
  const run = {
    trace: {
      tensors: Object.fromEntries(
        [...inputs, ...outputs].map((t) => [t.id, t]),
      ),
    },
  } as Run;
  return {
    run,
    op,
    trace: (index: number, id = outputs[0].id) =>
      traceCellContributors(run, op, id, index),
  };
}
const lesson = (data: object) => data as Operation["lesson"];
const relation = (rule: object) =>
  lesson({ interaction: "relation", relation: { operand: 0, ...rule } });
const cells = (result: ReturnType<typeof traceCellContributors>) =>
  result.sources.map((s) => [s.tensorId, s.index]);

describe("one-cell structural contributor tracing", () => {
  it("maps permutation and enormous reshapes without reading values", () => {
    const x = tensor("x", [1024, 1024, 768]);
    Object.defineProperty(x, "values", {
      get: () => {
        throw new Error("Do not scan tensor values");
      },
    });
    const reshaped = fixture("reshape", [x], [tensor("y", [1024, 768, 1024])], {
      lesson: lesson({ interaction: "mapping", mapping_rule: "identity" }),
    });
    expect(cells(reshaped.trace(x.numel - 1))).toEqual([["x", x.numel - 1]]);
    const transposed = fixture(
      "transpose",
      [tensor("x", [2, 3])],
      [tensor("y", [3, 2])],
      {
        lesson: lesson({
          interaction: "mapping",
          mapping_rule: "permutation",
          axis_order: [1, 0],
        }),
      },
    );
    expect(cells(transposed.trace(3))).toEqual([["x", 4]]);
  });

  it("locates the selected output of split/unbind and repeated join operands", () => {
    const split = fixture(
      "chunk",
      [tensor("x", [2, 5])],
      [tensor("a", [2, 3]), tensor("b", [2, 2])],
      { arguments: { dim: -1 } },
    );
    expect(cells(split.trace(3, "b"))).toEqual([["x", 9]]);
    const unbind = fixture(
      "unbind",
      [tensor("x", [2, 3])],
      [tensor("a", [2]), tensor("b", [2]), tensor("c", [2])],
      { arguments: { dim: 1 } },
    );
    expect(cells(unbind.trace(1, "c"))).toEqual([["x", 5]]);
    const a = tensor("a", [2, 2]);
    const joined = fixture("cat", [a, a], [tensor("out", [2, 4])], {
      arguments: { dim: 1 },
    });
    expect(joined.trace(7).sources).toEqual([
      { tensorId: "a", index: 3, role: "operand 2" },
    ]);
  });

  it("keeps two uses of the same element in x + x", () => {
    const x = tensor("x", [3]);
    const added = fixture("add", [x, x], [tensor("y", [3])], {
      lesson: relation({ rule: "elementwise", roles: ["input", "other"] }),
    });
    expect(added.trace(2)).toMatchObject({
      status: "mapped",
      total: 2,
      truncated: false,
    });
    expect(cells(added.trace(2))).toEqual([
      ["x", 2],
      ["x", 2],
    ]);
    expect(added.trace(2).sources.map((s) => s.role)).toEqual([
      "input",
      "other",
    ]);
  });

  it("uses trailing-axis broadcasting for elementwise operands", () => {
    const result = fixture(
      "mul",
      [tensor("a", [2, 1]), tensor("b", [3])],
      [tensor("y", [2, 3])],
      { lesson: relation({ rule: "elementwise", roles: ["input", "other"] }) },
    );
    expect(cells(result.trace(5))).toEqual([
      ["a", 1],
      ["b", 2],
    ]);
  });

  it("bounds billion-cell reductions while retaining exact total and first indices", () => {
    const size = 1_000_000_000;
    const x = tensor("x", [2, size]);
    Object.defineProperty(x, "values", {
      get: () => {
        throw new Error("No value access");
      },
    });
    const reduced = fixture("mean", [x], [tensor("y", [2])], {
      lesson: relation({ rule: "reduce", axes: [1], keepdim: false }),
    });
    const result = reduced.trace(1);
    expect(result).toMatchObject({
      status: "mapped",
      total: size,
      truncated: true,
    });
    expect(result.sources).toHaveLength(CELL_CONTRIBUTOR_LIMIT);
    expect(result.sources[0].index).toBe(size);
    expect(result.sources.at(-1)!.index).toBe(size + 63);
    expect(result.summary).toContain("first 64 of 1,000,000,000 source uses");
  });

  it("traces the first prefix entries rather than a tail sample", () => {
    const result = fixture(
      "cumsum",
      [tensor("x", [2, 1000])],
      [tensor("y", [2, 1000])],
      { lesson: relation({ rule: "prefix", axis: 1 }) },
    ).trace(1999);
    expect(result.total).toBe(1000);
    expect(result.sources[0].index).toBe(1000);
    expect(result.sources.at(-1)!.index).toBe(1063);
  });

  it("traces broadcast matrix multiplication exactly", () => {
    const result = fixture(
      "matmul",
      [tensor("a", [4, 1, 2, 3]), tensor("b", [5, 3, 2])],
      [tensor("y", [4, 5, 2, 2])],
    ).trace(79);
    expect(cells(result)).toEqual([
      ["a", 21],
      ["b", 25],
      ["a", 22],
      ["b", 27],
      ["a", 23],
      ["b", 29],
    ]);
    expect(result.total).toBe(6);
  });

  it("handles vector/matrix promotion and scalar dot products", () => {
    const vectorMatrix = fixture(
      "matmul",
      [tensor("a", [3]), tensor("b", [2, 3, 4])],
      [tensor("y", [2, 4])],
    );
    expect(cells(vectorMatrix.trace(6))).toEqual([
      ["a", 0],
      ["b", 14],
      ["a", 1],
      ["b", 18],
      ["a", 2],
      ["b", 22],
    ]);
    const x = tensor("x", [3]);
    const dot = fixture("matmul", [x, x], [tensor("y", [])]);
    expect(cells(dot.trace(0))).toEqual([
      ["x", 0],
      ["x", 0],
      ["x", 1],
      ["x", 1],
      ["x", 2],
      ["x", 2],
    ]);
    expect(dot.trace(0).total).toBe(6);
  });

  it("bounds billion-feature dot products independently of dimension size", () => {
    const size = 1_000_000_000;
    const result = fixture(
      "matmul",
      [tensor("a", [1, size]), tensor("b", [size, 1])],
      [tensor("y", [1, 1])],
    ).trace(0);
    expect(result).toMatchObject({
      status: "mapped",
      total: 2 * size,
      truncated: true,
    });
    expect(result.sources).toHaveLength(64);
    expect(cells(result).slice(-2)).toEqual([
      ["a", 31],
      ["b", 31],
    ]);
  });

  it("includes linear weights and the selected bias", () => {
    const result = fixture(
      "linear",
      [tensor("x", [2, 3]), tensor("w", [4, 3]), tensor("b", [4])],
      [tensor("y", [2, 4])],
    ).trace(6);
    expect(cells(result)).toEqual([
      ["x", 3],
      ["w", 6],
      ["x", 4],
      ["w", 7],
      ["x", 5],
      ["w", 8],
      ["b", 2],
    ]);
    expect(result.total).toBe(7);
  });

  it("counts valid convolution inputs and weights without inventing padding cells", () => {
    const result = fixture(
      "conv2d",
      [tensor("x", [1, 1, 2, 2]), tensor("w", [1, 1, 3, 3]), tensor("b", [1])],
      [tensor("y", [1, 1, 2, 2])],
      { arguments: { padding: 1 } },
    ).trace(0);
    expect(result).toMatchObject({
      status: "mapped",
      total: 14,
      truncated: false,
    });
    expect(
      result.sources.filter((s) => s.tensorId === "x").map((s) => s.index),
    ).toEqual([0, 1, 2, 3]);
    expect(
      result.sources.filter((s) => s.tensorId === "w").map((s) => s.index),
    ).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    expect(result.summary).toContain("padding is a constant");
  });

  it("bounds huge padded convolution kernels without scanning their padding", () => {
    const size = 1_000_000_001;
    const result = fixture(
      "conv1d",
      [tensor("x", [1, 1, 1]), tensor("w", [1, 1, size])],
      [tensor("y", [1, 1, 1])],
      { arguments: { padding: "same" } },
    ).trace(0);
    expect(result).toMatchObject({
      status: "mapped",
      total: size + 1,
      truncated: true,
    });
    expect(result.sources).toHaveLength(64);
    expect(result.sources.every((s) => s.tensorId === "w")).toBe(true);
  });

  it("uses the correct input channel group for grouped convolution", () => {
    const result = fixture(
      "conv1d",
      [tensor("x", [1, 4, 3]), tensor("w", [4, 2, 1])],
      [tensor("y", [1, 4, 3])],
      { arguments: { groups: 2 } },
    ).trace(7);
    expect(cells(result)).toEqual([
      ["x", 7],
      ["w", 4],
      ["x", 10],
      ["w", 5],
    ]);
  });

  it("describes pooling comparisons and omits constant border cells", () => {
    const max = fixture(
      "max_pool2d",
      [tensor("x", [1, 1, 2, 2])],
      [tensor("y", [1, 1, 2, 2])],
      { arguments: { kernel_size: 3, stride: 1, padding: 1 } },
    ).trace(0);
    expect(cells(max)).toEqual([
      ["x", 0],
      ["x", 1],
      ["x", 2],
      ["x", 3],
    ]);
    expect(max.total).toBe(4);
    expect(max.summary).toContain("compared");
    const adaptive = fixture(
      "adaptive_avg_pool2d",
      [tensor("x", [1, 1, 100000, 100000])],
      [tensor("y", [1, 1, 1, 1])],
      { arguments: { output_size: [1, 1] } },
    ).trace(0);
    expect(adaptive).toMatchObject({ total: 10_000_000_000, truncated: true });
    expect(adaptive.sources).toHaveLength(64);
    expect(adaptive.sources.at(-1)!.index).toBe(63);
  });

  it("maps softmax groups and layer norm affine parameters", () => {
    const softmax = fixture(
      "softmax",
      [tensor("x", [2, 3])],
      [tensor("y", [2, 3])],
      { arguments: { dim: -1 } },
    );
    expect(cells(softmax.trace(4))).toEqual([
      ["x", 3],
      ["x", 4],
      ["x", 5],
    ]);
    const norm = fixture(
      "layer_norm",
      [tensor("x", [2, 3]), tensor("w", [3]), tensor("b", [3])],
      [tensor("y", [2, 3])],
      {
        arguments: { normalized_shape: [3], weight: "tensor", bias: "tensor" },
      },
    );
    expect(cells(norm.trace(4))).toEqual([
      ["x", 3],
      ["x", 4],
      ["x", 5],
      ["w", 1],
      ["b", 1],
    ]);
    expect(norm.trace(4).total).toBe(5);
  });

  it("traces recorded embedding table lookups and constant padding", () => {
    const ids = tensor("indices", [2], [2, 0]);
    ids.dtype = "int64";
    const embedding = fixture(
      "embedding",
      [ids, tensor("table", [3, 2])],
      [tensor("y", [2, 2])],
      {
        lesson: lesson({
          interaction: "relation",
          relation: { rule: "table", operand: 1 },
          mapping: [4, 5, 0, 1],
        }),
      },
    );
    expect(cells(embedding.trace(1))).toEqual([
      ["indices", 0],
      ["table", 5],
    ]);
    const padded = fixture("pad", [tensor("x", [2])], [tensor("y", [4])], {
      lesson: relation({ rule: "pad", before: [1] }),
    });
    expect(padded.trace(0)).toMatchObject({
      status: "mapped",
      sources: [],
      total: 0,
    });
    expect(cells(padded.trace(2))).toEqual([["x", 1]]);
  });

  it("reads only the selected recorded mapping entry, even for a huge table", () => {
    const ids = tensor("indices", [1_000_000_000]);
    ids.dtype = "int64";
    const mapping = new Proxy([] as number[], {
      get(_target, key) {
        if (key === "length") return 2_000_000_000;
        if (key === "1999999999") return 5;
        throw new Error(`Unexpected table scan: ${String(key)}`);
      },
    });
    const result = fixture(
      "embedding",
      [ids, tensor("table", [3, 2])],
      [tensor("y", [1_000_000_000, 2])],
      {
        lesson: lesson({
          interaction: "relation",
          relation: { rule: "table", operand: 1 },
          mapping,
        }),
      },
    ).trace(1_999_999_999);
    expect(cells(result)).toEqual([
      ["indices", 999_999_999],
      ["table", 5],
    ]);
    expect(result).toMatchObject({
      status: "mapped",
      total: 2,
      truncated: false,
    });
  });

  it("includes the correct index cell for gather and index_select", () => {
    const index = tensor("ids", [2], [2, 0]);
    index.dtype = "int64";
    const selected = fixture(
      "index_select",
      [tensor("x", [2, 3]), index],
      [tensor("y", [2, 2])],
      {
        arguments: { dim: -1 },
        lesson: lesson({
          interaction: "relation",
          relation: { rule: "table", operand: 0 },
          mapping: [2, 0, 5, 3],
        }),
      },
    );
    expect(cells(selected.trace(3))).toEqual([
      ["ids", 1],
      ["x", 3],
    ]);
    const gatherIndices = tensor("ids", [2, 2], [2, 0, 1, 1]);
    gatherIndices.dtype = "int64";
    const gathered = fixture(
      "gather",
      [tensor("x", [2, 3]), gatherIndices],
      [tensor("y", [2, 2])],
      {
        arguments: { dim: 1 },
        lesson: lesson({
          interaction: "relation",
          relation: { rule: "table", operand: 0 },
          mapping: [2, 0, 4, 4],
        }),
      },
    );
    expect(cells(gathered.trace(2))).toEqual([
      ["ids", 2],
      ["x", 4],
    ]);
    gathered.op.lesson.mapping![2] = 1;
    expect(gathered.trace(2).status).toBe("unsupported");
  });

  it("does not present an unverified table selection as a complete dependency map", () => {
    const selected = fixture(
      "custom_selection",
      [tensor("x", [2]), tensor("indices", [2])],
      [tensor("y", [2])],
      {
        lesson: lesson({
          interaction: "relation",
          relation: { rule: "table", operand: 0 },
          mapping: [1, 0],
        }),
      },
    );
    expect(selected.trace(0).status).toBe("unsupported");
  });

  it("preserves repeated operands and summed terms for einsum", () => {
    const x = tensor("x", [1000]);
    const result = fixture("einsum", [x, x], [tensor("y", [])], {
      lesson: relation({
        rule: "einsum",
        inputs: ["i", "i"],
        output: "",
        sizes: { i: 1000 },
      }),
    }).trace(0);
    expect(result).toMatchObject({ total: 2000, truncated: true });
    expect(cells(result).slice(0, 4)).toEqual([
      ["x", 0],
      ["x", 0],
      ["x", 1],
      ["x", 1],
    ]);
    expect(result.sources).toHaveLength(64);
  });

  it("uses only the recorded chosen branch and requires its condition value", () => {
    const conditional = fixture(
      "where",
      [tensor("mask", [2], [0, 1]), tensor("x", [2]), tensor("z", [2])],
      [tensor("y", [2])],
      {
        lesson: relation({
          rule: "elementwise",
          roles: ["condition", "input", "other"],
        }),
      },
    );
    expect(cells(conditional.trace(0))).toEqual([
      ["mask", 0],
      ["z", 0],
    ]);
    expect(cells(conditional.trace(1))).toEqual([
      ["mask", 1],
      ["x", 1],
    ]);
    conditional.run.trace.tensors.mask.values = [];
    expect(conditional.trace(1).status).toBe("unsupported");
  });

  it("maps unfold and negative rolls using exact logical coordinates", () => {
    const unfolded = fixture(
      "unfold",
      [tensor("x", [6])],
      [tensor("y", [2, 3])],
      {
        arguments: { dimension: 0, size: 3, step: 2 },
        lesson: lesson({ interaction: "mapping", mapping_rule: "unfold" }),
      },
    );
    expect(cells(unfolded.trace(5))).toEqual([["x", 4]]);
    const rolled = fixture(
      "roll",
      [tensor("x", [2, 3])],
      [tensor("y", [2, 3])],
      {
        arguments: { dims: [-1], shifts: [-1] },
        lesson: lesson({ interaction: "mapping", mapping_rule: "roll" }),
      },
    );
    expect(cells(rolled.trace(2))).toEqual([["x", 0]]);
  });

  it("rejects invalid selections, missing tensors, inconsistent shapes and empty outputs", () => {
    const fixtureValue = fixture(
      "reshape",
      [tensor("x", [3])],
      [tensor("y", [3])],
      { lesson: lesson({ interaction: "mapping", mapping_rule: "identity" }) },
    );
    for (const index of [-1, 3, 0.5, Infinity])
      expect(fixtureValue.trace(index).status).toBe("invalid");
    expect(fixtureValue.trace(0, "x").status).toBe("invalid");
    fixtureValue.run.trace.tensors.x.numel = 2;
    expect(fixtureValue.trace(0).status).toBe("invalid");
    fixtureValue.op.inputs = ["missing"];
    expect(fixtureValue.trace(0).status).toBe("invalid");
    expect(
      fixture("relu", [tensor("x", [0])], [tensor("y", [0])]).trace(0).status,
    ).toBe("invalid");
  });

  it("declines failures, mutations, unknown operations and incompatible matmul batches", () => {
    const unknown = fixture("custom", [tensor("x", [2])], [tensor("y", [2])]);
    expect(unknown.trace(0).status).toBe("unsupported");
    unknown.op.status = "error";
    expect(unknown.trace(0).status).toBe("unsupported");
    unknown.op.status = "ok";
    unknown.op.arguments.out = "y";
    expect(unknown.trace(0).status).toBe("unsupported");
    const invalid = fixture(
      "bmm",
      [tensor("a", [1, 2, 3]), tensor("b", [5, 3, 2])],
      [tensor("y", [5, 2, 2])],
    );
    expect(invalid.trace(0).status).toBe("unsupported");
  });

  it("recognizes mutation alias snapshots as valid outputs without inventing their origins", () => {
    const mutated = fixture("add_", [tensor("x", [2])], [tensor("y", [2])], {
      mutations: [{ before: "x", after: "alias", kind: "alias" }],
    });
    mutated.run.trace.tensors.alias = tensor("alias", [2]);
    const result = mutated.trace(0, "alias");
    expect(result.status).toBe("unsupported");
    expect(result.summary).toContain("recorded write");
  });
});

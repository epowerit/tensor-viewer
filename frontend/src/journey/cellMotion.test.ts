import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import { CELL_MOTION_LIMIT, cellMotionPlan } from "./cellMotion";

const tensor = (id: string, shape: number[]): Tensor =>
  ({
    id,
    shape,
    numel: shape.reduce((a, b) => a * b, 1),
    dtype: "float32",
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
    arguments: { dim: -1 },
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
  return { run, op };
}

const lesson = (data: object) => data as Operation["lesson"];
const mapping = { interaction: "mapping", mapping_rule: "identity" };

describe("truthful whole-tensor motion on the journey canvas", () => {
  it("keeps repeated concat operands at distinct output positions", () => {
    const input = tensor("input", [2, 2]);
    const { run, op } = fixture(
      "cat",
      [input, input],
      [tensor("joined", [2, 4])],
    );
    const plan = cellMotionPlan(run, op);
    expect(plan.mode).toBe("assemble");
    expect(plan.movers.map((m) => m.sourceIndex)).toEqual([
      0, 1, 0, 1, 2, 3, 2, 3,
    ]);
    expect(plan.movers.map((m) => m.targetIndex)).toEqual([
      0, 1, 2, 3, 4, 5, 6, 7,
    ]);
    expect(
      plan.movers.every(
        (m) => m.sourceTensorId === "input" && m.targetTensorId === "joined",
      ),
    ).toBe(true);
  });

  it.each(["split", "chunk"])(
    "%s maps all outputs, including the last part",
    (kind) => {
      const { run, op } = fixture(
        kind,
        [tensor("x", [2, 3])],
        [tensor("left", [2, 2]), tensor("right", [2, 1])],
      );
      const plan = cellMotionPlan(run, op);
      expect(plan.mode).toBe("partition");
      expect(
        plan.movers.map((m) => [
          m.sourceIndex,
          m.targetTensorId,
          m.targetIndex,
        ]),
      ).toEqual([
        [0, "left", 0],
        [1, "left", 1],
        [2, "right", 0],
        [3, "left", 2],
        [4, "left", 3],
        [5, "right", 1],
      ]);
    },
  );

  it("stack and unbind preserve inserted-axis coordinates", () => {
    const parts = [tensor("a", [2]), tensor("b", [2])];
    const stack = fixture("stack", parts, [tensor("x", [2, 2])]);
    expect(
      cellMotionPlan(stack.run, stack.op).movers.map((m) => [
        m.sourceTensorId,
        m.sourceIndex,
        m.targetIndex,
      ]),
    ).toEqual([
      ["a", 0, 0],
      ["b", 0, 1],
      ["a", 1, 2],
      ["b", 1, 3],
    ]);
    const unbind = fixture("unbind", [tensor("x", [2, 2])], parts);
    expect(
      cellMotionPlan(unbind.run, unbind.op).movers.map((m) => [
        m.sourceIndex,
        m.targetTensorId,
        m.targetIndex,
      ]),
    ).toEqual([
      [0, "a", 0],
      [1, "b", 0],
      [2, "a", 1],
      [3, "b", 1],
    ]);
  });

  it("maps a transpose by the recorded permutation rather than flat identity", () => {
    const { run, op } = fixture(
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
    const plan = cellMotionPlan(run, op);
    expect(plan.mode).toBe("rearrange");
    expect(plan.movers.map((m) => m.sourceIndex)).toEqual([0, 3, 1, 4, 2, 5]);
  });

  it("reductions meet at their exact output group and retain every contributor", () => {
    const { run, op } = fixture(
      "sum",
      [tensor("x", [2, 3])],
      [tensor("y", [2])],
      {
        lesson: lesson({
          interaction: "relation",
          relation: { rule: "reduce", operand: 0, axes: [1], keepdim: false },
        }),
      },
    );
    const plan = cellMotionPlan(run, op);
    expect(plan.mode).toBe("collapse");
    expect(plan.movers.map((m) => [m.sourceIndex, m.targetIndex])).toEqual([
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 1],
      [4, 1],
      [5, 1],
    ]);
  });

  it("selection rules read the recorded operand and may reuse one source", () => {
    const { run, op } = fixture(
      "embedding",
      [tensor("indices", [3]), tensor("table", [2, 2])],
      [tensor("values", [3, 2])],
      {
        lesson: lesson({
          interaction: "relation",
          mapping: [2, 3, 0, 1, 2, 3],
          relation: { rule: "table", operand: 1 },
        }),
      },
    );
    const plan = cellMotionPlan(run, op);
    expect(plan.mode).toBe("select");
    expect(plan.movers.map((m) => m.sourceIndex)).toEqual([2, 3, 0, 1, 2, 3]);
    expect(plan.movers.every((m) => m.sourceTensorId === "table")).toBe(true);
  });

  it("declines large tensors without sampling or allocating their cells", () => {
    const input = tensor("x", [1024, 1024, 768]);
    Object.defineProperty(input, "values", {
      get() {
        throw new Error("Cell values must not be scanned");
      },
    });
    const { run, op } = fixture(
      "reshape",
      [input],
      [tensor("y", [1024, 768, 1024])],
      { lesson: lesson(mapping) },
    );
    const plan = cellMotionPlan(run, op);
    expect(plan.mode).toBe("none");
    expect(plan.movers).toEqual([]);
    expect(plan.caption).toContain("small tensors");
    const many = fixture(
      "cat",
      Array.from({ length: 1024 }, (_, i) => tensor(`empty${i}`, [0])),
      [tensor("empty", [0])],
    );
    expect(cellMotionPlan(many.run, many.op).mode).toBe("none");
  });

  it("caps the total scene and never truncates a mapping", () => {
    const a = tensor("a", [8, 8]);
    const accepted = fixture("cat", [a, a], [tensor("joined", [8, 16])]);
    expect(cellMotionPlan(accepted.run, accepted.op).movers).toHaveLength(
      CELL_MOTION_LIMIT,
    );
    const excessive = fixture(
      "cat",
      [a, a, tensor("one", [8, 1])],
      [tensor("joined", [8, 17])],
    );
    expect(cellMotionPlan(excessive.run, excessive.op).movers).toEqual([]);
  });

  it("never animates computations, unknown relations, mutations, or an arbitrary first output", () => {
    const { run, op } = fixture(
      "reshape",
      [tensor("x", [2, 2])],
      [tensor("y", [4])],
      { lesson: lesson(mapping) },
    );
    for (const patch of [
      { status: "error" },
      { kind: "linear" },
      { arguments: { out: "y" } },
      { mutations: [{ before: "x", after: "y", kind: "write" }] },
      { outputs: ["y", "y"] },
      {
        lesson: {
          interaction: "relation",
          relation: { rule: "unknown", operand: 0 },
        },
      },
      {
        lesson: {
          interaction: "relation",
          relation: { rule: "elementwise", operand: 0, roles: ["input"] },
        },
      },
    ])
      expect(cellMotionPlan(run, { ...op, ...patch } as Operation).mode).toBe(
        "none",
      );
  });

  it("rejects incomplete metadata and invalid index tables", () => {
    const { run, op } = fixture(
      "reshape",
      [tensor("x", [2, 2])],
      [tensor("y", [4])],
      { lesson: lesson(mapping) },
    );
    expect(cellMotionPlan(run, { ...op, inputs: ["missing"] }).mode).toBe(
      "none",
    );
    run.trace.tensors.x.numel = 3;
    expect(cellMotionPlan(run, op).mode).toBe("none");
    run.trace.tensors.x.numel = 4;
    op.lesson = lesson({
      interaction: "relation",
      mapping: [0, 1, 2, 100],
      relation: { rule: "table", operand: 0 },
    });
    expect(cellMotionPlan(run, op).mode).toBe("none");
  });
});

import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import cases from "./fixtures/layerNormalization.json";
import {
  layerNormalization,
  layerNormSelection,
  layerNormWindow,
  layerNormStatistics,
  layerNormCalculation,
} from "./layerNormalization";

function tensor(id: string, shape: number[], values: number[] = []): Tensor {
  return {
    id,
    name: id,
    shape,
    numel: shape.reduce((a, b) => a * b, 1),
    values,
    axes: shape.map((_, i) => `axis ${i}`),
    dtype: "float64",
    strides: [],
    storage_id: id,
    storage_offset: 0,
    contiguous: true,
    minimum: null,
    maximum: null,
    role: "intermediate",
    value_source: values.length ? "inline" : "shape",
  };
}
function fixture(index = 0) {
  const c = cases[index];
  const input = tensor("x", c.input.shape, c.input.values),
    output = tensor("y", c.output.shape, c.output.values);
  const tensors: Record<string, Tensor> = { x: input, y: output };
  const inputs = ["x"];
  if (c.weight) {
    tensors.w = tensor("w", c.weight.shape, c.weight.values);
    inputs.push("w");
  }
  if (c.bias) {
    tensors.b = tensor("b", c.bias.shape, c.bias.values);
    inputs.push("b");
  }
  const op: Operation = {
    id: "norm",
    index: 0,
    function: "torch.layer_norm",
    source: null,
    module: "",
    error: null,
    lesson: {
      title: "",
      summary: "",
      detail: "",
      category: "normalize",
      interaction: "layer_normalization",
      mapping: null,
      axis_order: null,
    },
    kind: "layer_norm",
    status: "ok",
    inputs,
    outputs: ["y"],
    arguments: { ...c.arguments },
  };
  const run = { trace: { tensors, operations: [op] } } as Run;
  return { run, op, p: layerNormalization(run, op)! };
}

describe("layer normalization", () => {
  cases.forEach((_, index) =>
    it(`reconstructs every PyTorch output and affine coordinate in case ${index + 1}`, () => {
      const { p } = fixture(index);
      expect(p).not.toBeNull();
      for (let i = 0; i < p.input.numel; i++) {
        const selected = layerNormSelection(p, i),
          samples = layerNormWindow(p, i, true);
        expect(samples).toHaveLength(p.size);
        expect(samples[0]).toBe(selected.group * p.size);
        const stats = layerNormStatistics(
          samples.map((j) => p.input.values[j]),
          p.size,
          p.eps,
        );
        const scale = p.weight ? p.weight.values[selected.feature] : 1,
          bias = p.bias ? p.bias.values[selected.feature] : 0;
        const calculation = layerNormCalculation(
          stats,
          p.input.values[i],
          scale,
          bias,
        )!;
        expect(calculation.output).toBeCloseTo(
          p.output.values[i] as number,
          10,
        );
        expect(selected.parameterCoordinates).toEqual(
          selected.coordinates.slice(-p.shape.length),
        );
      }
    }),
  );
  it("uses population variance, epsilon inside the root, and per-coordinate scale/bias", () => {
    const stats = layerNormStatistics([1, 3], 2, 3)!;
    expect(stats).toEqual({ mean: 2, variance: 1, denominator: 2 });
    expect(layerNormCalculation(stats, 3, 2, -4)).toEqual({
      centered: 1,
      normalized: 0.5,
      output: -3,
    });
  });
  it("identifies multi-axis groups rather than normalizing only rows", () => {
    const { p } = fixture(1);
    expect(p.shape).toEqual([2, 3]);
    expect(p.groups).toBe(2);
    expect(layerNormSelection(p, 10)).toEqual({
      group: 1,
      feature: 4,
      start: 6,
      coordinates: [1, 1, 1],
      prefix: [1],
      parameterCoordinates: [1, 1],
    });
    expect(layerNormWindow(p, 10, true)).toEqual([6, 7, 8, 9, 10, 11]);
    const whole = fixture(5).p;
    expect(whole.groups).toBe(1);
    expect(layerNormSelection(whole, 5).prefix).toEqual([]);
  });
  it("keeps logical coordinates independent of non-contiguous storage", () => {
    const { run, op, p } = fixture();
    p.input.contiguous = false;
    p.input.strides = [4, 8, 1];
    p.input.storage_offset = 37;
    expect(layerNormalization(run, op)).not.toBeNull();
    expect(layerNormSelection(p, 23).coordinates).toEqual([1, 2, 3]);
    expect(layerNormWindow(p, 23, true)).toEqual([20, 21, 22, 23]);
  });
  it("distinguishes bias-only calls and accepts only unambiguous legacy roles", () => {
    const biasOnly = fixture(3);
    expect(biasOnly.p.weight).toBeUndefined();
    expect(biasOnly.p.bias?.id).toBe("b");
    const { run, op } = fixture(2);
    delete op.arguments.weight;
    expect(layerNormalization(run, op)?.weight?.id).toBe("w");
    delete op.arguments.bias;
    expect(layerNormalization(run, op)).toBeNull();
    const both = fixture();
    delete both.op.arguments.weight;
    delete both.op.arguments.bias;
    expect(layerNormalization(both.run, both.op)?.bias?.id).toBe("b");
    const neither = fixture(4);
    delete neither.op.arguments.weight;
    delete neither.op.arguments.bias;
    expect(layerNormalization(neither.run, neither.op)?.weight).toBeUndefined();
  });
  it("bounds billion-element group navigation and never uses a partial mean", () => {
    const { run, op } = fixture(4);
    run.trace.tensors.x = tensor("x", [1024, 1024, 1024]);
    run.trace.tensors.y = tensor("y", [1024, 1024, 1024]);
    op.arguments.normalized_shape = [1024];
    const p = layerNormalization(run, op)!;
    const selected = p.input.numel - 1;
    expect(layerNormSelection(p, selected)).toMatchObject({
      group: 1048575,
      feature: 1023,
      prefix: [1023, 1023],
    });
    expect(layerNormWindow(p, selected, true)).toHaveLength(8);
    expect(layerNormWindow(p, selected).at(-1)).toBe(selected);
    expect(
      layerNormStatistics([1, 2, 3, 4, 5, 6, 7, 8], 1024, p.eps),
    ).toBeNull();
    run.trace.tensors.x = tensor("x", [1, 2 ** 39, 2]);
    run.trace.tensors.y = tensor("y", [1, 2 ** 39, 2]);
    op.arguments.normalized_shape = [2 ** 39, 2];
    const huge = layerNormalization(run, op)!;
    expect(layerNormWindow(huge, 2 ** 40 - 1, true)).toHaveLength(8);
    expect(layerNormSelection(huge, 2 ** 40 - 1).parameterCoordinates).toEqual([
      2 ** 39 - 1,
      1,
    ]);
  });
  it("handles constant groups without inventing finite results at zero epsilon", () => {
    for (const value of [0.1, 1e300])
      expect(layerNormStatistics(Array(255).fill(value), 255, 0)).toEqual({
        mean: value,
        variance: 0,
        denominator: 0,
      });
    const stats = layerNormStatistics([3, 3, 3], 3, 0)!;
    expect(stats).toEqual({ mean: 3, variance: 0, denominator: 0 });
    expect(layerNormCalculation(stats, 3, 1, 0)).toBeNull();
    const stabilized = layerNormStatistics([3], 1, 1e-5)!;
    expect(layerNormCalculation(stabilized, 3, 2, 5)?.output).toBe(5);
  });
  it("rejects missing, non-finite, overflowing values and incomplete parameters", () => {
    for (const values of [
      [1, undefined],
      [1, "nan"],
      [1, Infinity],
      [NaN, 0],
      [1e308, -1e308],
    ])
      expect(layerNormStatistics(values, 2, 1e-5)).toBeNull();
    expect(layerNormStatistics([], 0, 1e-5)).toBeNull();
    const stats = layerNormStatistics([1, 3], 2, 1e-5)!;
    expect(layerNormCalculation(stats, undefined, 1, 0)).toBeNull();
    expect(
      layerNormCalculation(stats, 3, undefined, 0)?.output,
    ).toBeUndefined();
    expect(layerNormCalculation(stats, 3, 1, "nan")?.output).toBeUndefined();
    expect(
      layerNormCalculation(stats, 3, 1e308, 1e308)?.output,
    ).toBeUndefined();
  });
  it("rejects failed, mutated, empty, mismatched, and ambiguous traces", () => {
    const { run, op } = fixture();
    for (const arguments_ of [
      { ...op.arguments, normalized_shape: [3] },
      { ...op.arguments, eps: null },
      { ...op.arguments, eps: -1 },
      { ...op.arguments, eps: "nan" },
      { ...op.arguments, weight: null },
      { ...op.arguments, weight: "unknown" },
    ])
      expect(
        layerNormalization(run, { ...op, arguments: arguments_ }),
      ).toBeNull();
    expect(layerNormalization(run, { ...op, status: "error" })).toBeNull();
    expect(
      layerNormalization(run, {
        ...op,
        mutations: [{ before: "x", after: "y", kind: "write" }],
      }),
    ).toBeNull();
    expect(layerNormalization(run, { ...op, inputs: ["missing"] })).toBeNull();
    run.trace.tensors.w.dtype = "float32";
    expect(layerNormalization(run, op)).toBeNull();
    run.trace.tensors.w.dtype = "float64";
    run.trace.tensors.b.shape = [2, 2];
    expect(layerNormalization(run, op)).toBeNull();
    run.trace.tensors.b.shape = [4];
    run.trace.tensors.y.shape = [3, 2, 4];
    expect(layerNormalization(run, op)).toBeNull();
    run.trace.tensors.y.shape = [2, 3, 4];
    run.trace.tensors.x.numel = 0;
    expect(layerNormalization(run, op)).toBeNull();
  });
});

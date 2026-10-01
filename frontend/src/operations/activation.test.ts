import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import cases from "./fixtures/activations.json";
import { ActivationView } from "./ActivationView";
import {
  activationPlot,
  activationValue,
  activationWindow,
  tensorActivation,
  type Activation,
} from "./activation";

function fixture(index = 0) {
  const c = cases[index];
  const tensor = (id: string, values: number[]): Tensor => ({
    id,
    name: id,
    shape: [1, 1, values.length],
    numel: values.length,
    values: [...values],
    axes: ["batch", "tokens", "features"],
    dtype: "float64",
    strides: [values.length, values.length, 1],
    storage_id: id,
    storage_offset: 0,
    contiguous: true,
    minimum: null,
    maximum: null,
    role: "intermediate",
    value_source: "inline",
  });
  const op: Operation = {
    id: "op0",
    index: 0,
    kind: c.kind,
    function: `torch.${c.kind}`,
    module: "",
    inputs: ["x"],
    outputs: ["y"],
    arguments: { approximate: c.approximate },
    status: "ok",
    source: null,
    error: null,
    lesson: {
      title: "",
      summary: "",
      detail: "",
      category: "compute",
      interaction: "activation",
      mapping: null,
      axis_order: null,
    },
  };
  const run: Run = {
    id: "run1",
    project_id: "project1",
    created_at: "2026-09-29T00:00:00Z",
    project: {
      name: "Activations",
      code: "",
      class_name: "Example",
      constructor: {},
      input: {
        shape: [1, 1, c.input.length],
        dtype: "float64",
        generator: "arange",
        axis_names: [],
        seed: 7,
      },
    },
    trace: {
      schema_version: "1",
      input_ids: ["x"],
      output_ids: ["y"],
      error: null,
      stdout: "",
      duration_ms: 0,
      tensors: { x: tensor("x", c.input), y: tensor("y", c.output) },
      operations: [op],
    },
  };
  return { run, op, p: tensorActivation(run, op)! };
}

describe("elementwise activations", () => {
  cases.forEach((c, index) =>
    it(`matches captured PyTorch ${c.kind} (${c.approximate}) across zero and the tails`, () => {
      const { p } = fixture(index);
      expect(p).not.toBeNull();
      for (let i = 0; i < c.input.length; i++) {
        const estimate = activationValue(p, c.input[i])!;
        expect(Number.isFinite(estimate)).toBe(true);
        // GELU's Gaussian CDF reference uses an explicitly approximate erf.
        expect(Math.abs(estimate - c.output[i])).toBeLessThan(
          c.kind === "gelu" && c.approximate === "none" ? 3e-7 : 1e-12,
        );
      }
    }),
  );

  it("keeps GELU modes distinct and explains negative attenuation rather than clipping", () => {
    const exact = fixture(1).p,
      tanh = fixture(2).p;
    expect(activationValue(exact, -1)).toBeLessThan(0);
    expect(
      Math.abs(activationValue(exact, 2)! - activationValue(tanh, 2)!),
    ).toBeGreaterThan(1e-5);
    expect(activationValue(fixture(0).p, -1)).toBe(0);
    expect(activationValue(fixture(3).p, 0)).toBe(0.5);
    expect(activationValue(fixture(4).p, 0)).toBe(0);
  });

  it("handles extreme finite values without overflowing the reference and never invents non-finite inputs", () => {
    for (let i = 0; i < cases.length; i++) {
      const { p } = fixture(i);
      for (const value of [-Number.MAX_VALUE, Number.MAX_VALUE])
        expect(Number.isFinite(activationValue(p, value))).toBe(true);
      for (const value of [
        undefined,
        "nan",
        "inf",
        "9007199254740993",
        NaN,
        Infinity,
        -Infinity,
      ])
        expect(activationValue(p, value)).toBeUndefined();
    }
  });

  it("supports scalar and non-contiguous logical coordinates without building a large map", () => {
    const { run, op } = fixture();
    for (const tensor of Object.values(run.trace.tensors)) {
      tensor.shape = [];
      tensor.numel = 1;
      tensor.values = [0];
    }
    expect(tensorActivation(run, op)).not.toBeNull();
    for (const tensor of Object.values(run.trace.tensors)) {
      tensor.shape = [1024, 1024, 1024];
      tensor.numel = 2 ** 30;
      tensor.values = [];
      tensor.value_source = "shape";
      tensor.contiguous = false;
      tensor.strides = [1, 1024, 1024 * 1024];
      tensor.storage_offset = 500;
    }
    expect(tensorActivation(run, op)).not.toBeNull();
    expect(activationWindow(2 ** 30, 2 ** 30 - 1)).toHaveLength(8);
    expect(activationWindow(2 ** 30, 2 ** 30 - 1).at(-1)).toBe(2 ** 30 - 1);
    expect(activationWindow(11, 10)).toEqual([8, 9, 10]);
    expect(activationWindow(1, 0)).toEqual([0]);
  });

  it("validates the recorded operation, shape, dtype, approximation, and mutation state", () => {
    const { run, op } = fixture(1);
    for (const changes of [
      { kind: "softmax" },
      { status: "error" },
      { inputs: ["missing"] },
      { inputs: ["x", "x"] },
      { outputs: [] },
      { mutations: [{ before: "x", after: "y", kind: "write" }] },
      { arguments: { approximate: "unknown" } },
      { arguments: { approximate: null } },
      { arguments: { inplace: true } },
      { arguments: { out: "tensor" } },
    ])
      expect(
        tensorActivation(run, { ...op, ...changes } as Operation),
      ).toBeNull();
    const input = run.trace.tensors.x,
      output = run.trace.tensors.y;
    input.dtype = output.dtype = "int64";
    expect(tensorActivation(run, op)).toBeNull();
    expect(tensorActivation(run, { ...op, kind: "relu" })).not.toBeNull();
    input.dtype = output.dtype = "float64";
    output.shape = [1, 57, 1];
    expect(tensorActivation(run, op)).toBeNull();
    output.shape = input.shape;
    output.numel = 56;
    expect(tensorActivation(run, op)).toBeNull();
  });
});

describe("activation chart and presentation", () => {
  it("plots recorded values, even when they differ from reference arithmetic", () => {
    const p = fixture(1).p;
    const plot = activationPlot(
      p,
      [{ index: 3, input: -1, output: -0.2 }],
      false,
    );
    expect(plot.points[0]).toMatchObject({
      index: 3,
      input: -1,
      output: -0.2,
      ...plot.project(-1, -0.2),
    });
    expect(plot.samples.length).toBeLessThanOrEqual(256);
    expect(plot.omitted).toBe(0);
  });
  it("identifies excluded cells and fits the visible window without a tensor-wide scan", () => {
    const p = fixture().p;
    const pairs = [
      { index: 0, input: -1000, output: 0 },
      { index: 1, input: 1000, output: 1000 },
      { index: 2, input: "nan", output: "nan" },
    ];
    const near = activationPlot(p, pairs, false),
      fitted = activationPlot(p, pairs, true);
    expect(near.points).toHaveLength(0);
    expect(near.omitted).toBe(2);
    expect(fitted.points).toHaveLength(2);
    expect(fitted.omitted).toBe(0);
    expect([fitted.xmin, fitted.xmax]).toEqual([-1000, 1000]);
    expect(fitted.samples).toContainEqual(fitted.project(0, 0));
  });
  it("keeps chart geometry finite at opposite floating-point extremes", () => {
    for (let i = 0; i < cases.length; i++) {
      const p = fixture(i).p;
      const pairs = [-Number.MAX_VALUE, Number.MAX_VALUE].map(
        (input, index) => ({ index, input, output: activationValue(p, input) }),
      );
      const plot = activationPlot(p, pairs, true);
      for (const point of [...plot.samples, ...plot.points]) {
        expect(Number.isFinite(point.x) && Number.isFinite(point.y)).toBe(true);
        expect(point.x).toBeGreaterThanOrEqual(58);
        expect(point.x).toBeLessThanOrEqual(606);
        expect(point.y).toBeGreaterThanOrEqual(36 - 1e-10);
        expect(point.y).toBeLessThanOrEqual(250 + 1e-10);
      }
    }
  });
  it("distinguishes reference curves, captured results, and shapes-only runs", () => {
    const { run, p } = fixture(1);
    p.input.values[0] = p.output.values[0] = 0;
    const render = (activation: Activation, showValues: boolean) =>
      renderToStaticMarkup(
        createElement(ActivationView, {
          activation,
          run,
          showValues,
          onShowValues: () => {},
          onDetails: () => {},
        }),
      );
    const numeric = render(p, true);
    expect(numeric).toContain("Recorded output");
    expect(numeric).toContain("Line: GELU reference");
    expect(numeric).toContain(
      'role="button" tabindex="0" aria-label="Select activation cell',
    );
    expect(numeric).toContain("finite captured pairs are");
    const hidden = render(p, false);
    expect(hidden).toContain("Values are hidden");
    expect(hidden).not.toContain('aria-label="Select activation cell');
    p.input.value_source = p.output.value_source = "shape";
    const shape = render(p, true);
    expect(shape).toContain("This run captured shapes only");
    expect(shape).not.toContain('aria-label="Select activation cell');
  });
});

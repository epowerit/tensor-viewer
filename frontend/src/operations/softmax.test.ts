import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import cases from "./fixtures/softmax.json";
import { SoftmaxView } from "./SoftmaxView";
import {
  softmaxCalculation,
  softmaxIndex,
  softmaxSelection,
  softmaxSummary,
  softmaxWindow,
  tensorSoftmax,
} from "./softmax";
import { product, unravel } from "../tensors/coordinates";

function fixture(index = 0) {
  const c = cases[index];
  const tensor = (id: string, values: (number | string)[]): Tensor => ({
    id,
    name: id,
    shape: [...c.shape],
    numel: product(c.shape),
    values: [...values],
    axes: c.shape.map((_, i) => `axis${i}`),
    dtype: "float64",
    strides: [],
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
    kind: "softmax",
    function: "torch.softmax",
    module: "",
    inputs: ["x"],
    outputs: ["y"],
    arguments: { dim: c.dim },
    status: "ok",
    source: null,
    error: null,
    lesson: {
      title: "",
      summary: "",
      detail: "",
      category: "normalize",
      interaction: "normalization",
      mapping: null,
      axis_order: null,
    },
  };
  const run: Run = {
    id: "run1",
    project_id: "project1",
    created_at: "2026-09-30T00:00:00Z",
    project: {
      name: "Softmax",
      code: "",
      class_name: "Example",
      constructor: {},
      input: {
        shape: [2, 3, 4],
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
  return { run, op, p: tensorSoftmax(run, op)! };
}

describe("softmax geometry and arithmetic", () => {
  cases.forEach((c, i) =>
    it(`follows native PyTorch ${c.name}`, () => {
      const { p } = fixture(i);
      expect(p).not.toBeNull();
      for (let selected = 0; selected < p.input.numel; selected++) {
        const { group, member } = softmaxSelection(p, selected);
        expect(softmaxIndex(p, group, member)).toBe(selected);
        const full = softmaxWindow(p, selected, true);
        expect(full).toHaveLength(p.size);
        const summary = softmaxSummary(
          full.map((s) => c.input[s.index]),
          p.size,
        );
        const calc = softmaxCalculation(summary, c.input[selected]);
        if (typeof c.output[selected] === "number")
          expect(calc?.weight).toBeCloseTo(c.output[selected] as number, 13);
        else expect(calc).toBeNull();
        const fixed = unravel(selected, p.input.shape);
        for (const cell of full) {
          const coordinates = unravel(cell.index, p.input.shape);
          coordinates.forEach((n, axis) => {
            if (axis !== p.axis) expect(n).toBe(fixed[axis]);
          });
        }
      }
    }),
  );
  it("keeps denominators unchanged by adding a common offset and handles masks explicitly", () => {
    const first = softmaxSummary([-1, 0, 1, "-inf"], 4)!;
    const shifted = softmaxSummary([999, 1000, 1001, "-inf"], 4)!;
    expect(first.denominator).toBe(shifted.denominator);
    expect(softmaxCalculation(first, "-inf")).toEqual({
      shifted: -Infinity,
      exponential: 0,
      weight: 0,
    });
    expect(softmaxSummary(["-inf", "-inf"], 2)).toEqual({
      status: "all_masked",
      masked_count: 2,
    });
    for (const bad of ["nan", "inf", NaN, Infinity])
      expect(softmaxSummary([0, bad], 2)?.status).toBe("non_finite");
    for (const values of [[0], [0, undefined], [0, "corrupt"]])
      expect(softmaxSummary(values, 2)).toBeNull();
    expect(softmaxCalculation(first, 2)).toBeNull();
  });
  it("reaches every axis group without allocating a large mapping", () => {
    const { run, op } = fixture();
    for (const t of Object.values(run.trace.tensors)) {
      t.shape = [1024, 1024, 1024];
      t.numel = 2 ** 30;
      t.value_source = "shape";
      t.values = [];
      t.contiguous = false;
      t.strides = [1, 1024, 1024 ** 2];
    }
    for (const dim of [0, 1, 2, -1]) {
      const p = tensorSoftmax(run, { ...op, arguments: { dim } })!;
      for (const selected of [0, 12943826, 2 ** 30 - 1]) {
        const { group, member } = softmaxSelection(p, selected);
        expect(softmaxIndex(p, group, member)).toBe(selected);
        const window = softmaxWindow(p, selected, true);
        expect(window).toHaveLength(8);
        expect(window.some((s) => s.index === selected)).toBe(true);
      }
    }
  });
  it("rejects unverified geometry, dtype conversions, implicit axes, and mutations", () => {
    const { run, op } = fixture();
    for (const change of [
      { kind: "log_softmax" },
      { status: "error" },
      { inputs: [] },
      { inputs: ["missing"] },
      { outputs: ["y", "y"] },
      { arguments: {} },
      { arguments: { dim: null } },
      { arguments: { dim: true } },
      { arguments: { dim: 3 } },
      { arguments: { dim: -4 } },
      { arguments: { dim: 1.5 } },
      { arguments: { dim: 1, dtype: "torch.float32" } },
      { arguments: { dim: 1, out: "tensor" } },
      { mutations: [{ before: "x", after: "y", kind: "write" }] },
    ])
      expect(tensorSoftmax(run, { ...op, ...change } as Operation)).toBeNull();
    for (const change of [
      { shape: [4, 3, 2] },
      { dtype: "float32" },
      { numel: 0 },
      { numel: 25 },
    ]) {
      const altered = {
        ...run,
        trace: {
          ...run.trace,
          tensors: {
            ...run.trace.tensors,
            y: { ...run.trace.tensors.y, ...change },
          },
        },
      };
      expect(tensorSoftmax(altered, op)).toBeNull();
    }
  });
});

describe("softmax presentation", () => {
  const render = (index: number, showValues = true, shape = false) => {
    const { run, p } = fixture(index);
    if (shape) {
      p.input.value_source = p.output.value_source = "shape";
      p.input.values = p.output.values = [];
    }
    return renderToStaticMarkup(
      createElement(SoftmaxView, {
        softmax: p,
        run,
        showValues,
        onShowValues: () => {},
        onDetails: () => {},
      }),
    );
  };
  it("separates reference stages from recorded weights and keeps each score interactive", () => {
    const markup = render(0);
    for (const text of [
      "Reference calculation",
      "01 · Find the maximum",
      "02 · Shift the score",
      "03 · Exponentiate",
      "04 · Normalize",
      "Recorded PyTorch output",
      "Softmax group",
      "Softmax score",
      "Inspect softmax score 3",
    ])
      expect(markup).toContain(text);
    expect(markup).toContain("softmax-weight-bar");
    expect(render(5)).toContain("negative-infinity scores contribute zero");
    expect(render(6)).toContain("NaN or positive infinity");
  });
  it("keeps structural exploration while hiding all numeric calculations and bars", () => {
    expect(render(0, false)).toContain("Values hidden");
    expect(render(0, false)).not.toContain("softmax-weight-bar");
    expect(render(0, true, true)).toContain("This run captured shapes only");
    expect(render(0, true, true)).not.toContain("softmax-weight-bar");
  });
});

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import { product, unravel } from "../tensors/coordinates";
import cases from "./fixtures/reductions.json";
import { ReductionView } from "./ReductionView";
import {
  reductionInputSelection,
  reductionReference,
  reductionSelection,
  reductionWindow,
  tensorReduction,
} from "./reduction";

function fixture(index = 0) {
  const c = cases[index];
  const tensor = (
    id: string,
    shape: number[],
    values: (number | string)[],
    dtype: string,
  ): Tensor => ({
    id,
    name: id,
    shape: [...shape],
    values: [...values],
    dtype,
    numel: product(shape),
    axes: shape.map((_, i) => `axis ${i}`),
    strides: shape.map((_, i) => product(shape.slice(i + 1))),
    storage_id: id,
    storage_offset: 0,
    contiguous: true,
    role: "intermediate",
    value_source: "inline",
    minimum: null,
    maximum: null,
  });
  const op: Operation = {
    id: "op0",
    index: 0,
    kind: c.kind,
    function: `torch.${c.kind}`,
    module: "",
    inputs: ["x"],
    outputs: ["y"],
    arguments: { ...c.arguments },
    status: "ok",
    source: null,
    error: null,
    lesson: {
      title: "",
      summary: "",
      detail: "",
      category: "compute",
      interaction: "reduction",
      mapping: null,
      axis_order: null,
    },
  };
  const run: Run = {
    id: "run1",
    project_id: "project1",
    created_at: "2026-09-30T00:00:00Z",
    project: {
      name: "Reductions",
      code: "",
      class_name: "Example",
      constructor: {},
      input: {
        shape: [...c.shape],
        dtype: "float64",
        generator: "arange",
        seed: 7,
        axis_names: [],
      },
    },
    trace: {
      schema_version: "1",
      input_ids: ["x"],
      output_ids: ["y"],
      error: null,
      stdout: "",
      duration_ms: 0,
      tensors: {
        x: tensor("x", c.shape, c.input, c.dtype),
        y: tensor("y", c.output_shape, c.output, c.output_dtype),
      },
      operations: [op],
    },
  };
  return { run, op, p: tensorReduction(run, op)! };
}

describe("mean and sum reduction groups", () => {
  cases.forEach((c, i) =>
    it(`follows every input and reproduces native PyTorch ${c.name}`, () => {
      const { p } = fixture(i);
      expect(p).not.toBeNull();
      expect(p.input.numel).toBe(p.output.numel * p.size);
      for (let output = 0; output < p.output.numel; output++) {
        const outputCoordinates = unravel(output, p.output.shape);
        // Independent enumeration: match only the non-reduced coordinates.
        const kept = p.input.shape
          .map((_, axis) => axis)
          .filter((axis) => !p.axes.includes(axis));
        const expected = Array.from(
          { length: p.input.numel },
          (_, input) => input,
        ).filter((input) =>
          kept.every(
            (axis, position) =>
              unravel(input, p.input.shape)[axis] ===
              outputCoordinates[p.keepdim ? axis : position],
          ),
        );
        const group = reductionWindow(p, output, 0, true);
        expect(group.map((s) => s.input)).toEqual(expected);
        group.forEach((s) => {
          expect(reductionInputSelection(p, s.input)).toEqual({
            output,
            term: s.term,
          });
          expect(reductionSelection(p, output, s.term)).toEqual(s);
        });
        const reference = reductionReference(
          p,
          group.map((s) => p.input.values[s.input]),
        );
        if (p.cast) expect(reference).toBeNull();
        else if (typeof c.output[output] === "string")
          expect(reference?.result).toBe(c.output[output]);
        else
          expect(reference?.result).toBeCloseTo(c.output[output] as number, 12);
      }
    }),
  );

  it("does not assume adjacent reduced axes or contiguous physical storage", () => {
    const { run, op, p } = fixture(2);
    p.input.strides = [1, 8, 2];
    p.input.contiguous = false;
    p.input.storage_offset = 100;
    expect(tensorReduction(run, op)).not.toBeNull();
    expect(reductionWindow(p, 1, 0, true).map((s) => s.input)).toEqual([
      4, 5, 6, 7, 16, 17, 18, 19,
    ]);
    expect(reductionInputSelection(p, 19)).toEqual({ output: 1, term: 7 });
  });

  it("bounds billion-element groups and reaches their final contributor", () => {
    const { run, op } = fixture(2);
    Object.assign(run.trace.tensors.x, {
      shape: [1024, 1024, 1024],
      numel: 2 ** 30,
      values: [],
      value_source: "shape",
    });
    Object.assign(run.trace.tensors.y, {
      shape: [1024],
      numel: 1024,
      values: [],
      value_source: "shape",
    });
    const p = tensorReduction(run, op)!;
    expect(p.size).toBe(2 ** 20);
    expect(reductionInputSelection(p, 2 ** 30 - 1)).toEqual({
      output: 1023,
      term: 2 ** 20 - 1,
    });
    const group = reductionWindow(p, 1023, p.size - 1, true);
    expect(group).toHaveLength(8);
    expect(group.at(-1)?.inputCoordinates).toEqual([1023, 1023, 1023]);
    expect(reductionReference(p, Array(8).fill(1))).toBeNull();
  });

  it("includes every term of small groups, and only the final window of large uneven groups", () => {
    const { run, op } = fixture(0);
    for (const size of [1, 3, 8, 9, 255, 256, 257, 1027]) {
      Object.assign(run.trace.tensors.x, {
        shape: [2, size, 4],
        numel: 8 * size,
      });
      const p = tensorReduction(run, op)!;
      const window = reductionWindow(p, 7, size - 1);
      expect(window.length).toBeLessThanOrEqual(8);
      expect(window.at(-1)?.term).toBe(size - 1);
      expect(reductionWindow(p, 7, size - 1, true)).toHaveLength(
        size <= 256 ? size : window.length,
      );
    }
  });

  it("never substitutes missing, partial, nonfinite, cast, or overflowing arithmetic", () => {
    const { p } = fixture(0);
    expect(reductionReference(p, [1, 2])).toBeNull();
    for (const missing of [undefined, NaN, Infinity, "nan", "inf"])
      expect(reductionReference(p, [1, missing, 3])).toBeNull();
    expect(reductionReference({ ...p, cast: true }, [1, 2, 3])).toBeNull();
    expect(reductionReference(p, [1e308, 1e308, 1e308])?.result).toBeCloseTo(
      1e308,
      -294,
    );
    expect(
      reductionReference({ ...p, kind: "sum" }, [1e308, 1e308, 1e308]),
    ).toBeNull();
    expect(
      reductionReference({ ...p, kind: "sum" }, [1e16, 1, -1e16])?.result,
    ).toBe(1);
    const integer = fixture(10).p;
    expect(reductionReference(integer, [9007199254740992, 2])).toBeNull();
    expect(reductionReference(integer, ["9007199254740993", 2])?.result).toBe(
      "9007199254740995",
    );
    expect(
      reductionReference(integer, [
        "-9223372036854775808",
        "-9223372036854775808",
      ])?.result,
    ).toBe("-18446744073709551616");
  });

  it("rejects ambiguous or inconsistent imported recordings", () => {
    const { run, op } = fixture(0);
    for (const changes of [
      { kind: "max" },
      { status: "error" },
      { inputs: ["absent"] },
      { inputs: ["x", "x"] },
      { outputs: [] },
      { mutations: [{ before: "x", after: "y", kind: "write" }] },
      ...[
        { dim: [1, -2] },
        { dim: 3 },
        { dim: true },
        { dim: "tokens" },
        { dim: 1.5 },
        { dim: [null] },
        { dim: 1, keepdim: null },
        { dim: 1, keepdim: 1 },
        { dim: 1, dtype: "float64" },
        { dim: 1, dtype: "torch.int64" },
        { dim: 1, out: "tensor" },
        { dim: 0 },
      ].map((arguments_) => ({ arguments: arguments_ })),
    ])
      expect(
        tensorReduction(run, { ...op, ...changes } as Operation),
      ).toBeNull();
    for (const changes of [
      { shape: [4, 2] },
      { shape: [0, 4], numel: 0 },
      { numel: 0 },
      { dtype: "complex64" },
      { shape: [2, 4.5], numel: 9 },
    ]) {
      const changed = {
        ...run,
        trace: {
          ...run.trace,
          tensors: {
            ...run.trace.tensors,
            y: { ...run.trace.tensors.y, ...changes },
          },
        },
      };
      expect(tensorReduction(changed, op)).toBeNull();
    }
  });

  it("renders bounded, labeled structural lessons without invented numbers", () => {
    const { run, p } = fixture(0);
    p.input.value_source = p.output.value_source = "shape";
    const render = (showValues: boolean) =>
      renderToStaticMarkup(
        createElement(ReductionView, {
          reduction: p,
          run,
          showValues,
          onShowValues: () => {},
          onDetails: () => {},
        }),
      );
    const html = render(true);
    expect(html).toContain('aria-label="Mean reduction lesson"');
    expect(html).toContain("Shapes only · no numeric values");
    expect(html).toContain('aria-label="Reduction contributor"');
    expect(html).toContain("y = S / 3");
    expect(html).toContain("Reduced axes disappear");
    expect(html).not.toContain("later</small>");
    expect(html.length).toBeLessThan(150000);
    p.input.value_source = p.output.value_source = "inline";
    expect(render(false)).toContain("Values hidden");
    expect(render(true)).toContain("Recorded PyTorch output");
  });

  it("shows scalar arithmetic without imaginary contributors and keeps output rounding visible", () => {
    const render = (index: number, output?: number) => {
      const { run, p } = fixture(index);
      if (output !== undefined) p.output.values[0] = output;
      return renderToStaticMarkup(
        createElement(ReductionView, {
          reduction: p,
          run,
          showValues: true,
          onShowValues: () => {},
          onDetails: () => {},
        }),
      );
    };
    const scalar = render(6);
    expect(scalar).toContain("S = x₀");
    expect(scalar).not.toContain("x₁");
    expect(scalar).toContain("1 independent group");
    // Long decimals must remain distinct from their integer reference, not both round to 1.6e6.
    expect(render(0, 1567102.875)).toContain(">1567102.875</strong>");
  });
});

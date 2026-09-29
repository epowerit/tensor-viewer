import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import fixtures from "./fixtures/convolutions.json";
import { product, ravel, unravel } from "../tensors/coordinates";
import { contributionSum, finiteProduct } from "./linear";
import {
  tensorConvolution,
  convolutionSelection,
  convolutionInputSelection,
  convolutionWeightSelection,
  convolutionWindow,
  nearestConvolutionSample,
} from "./convolution";

function tensor(id: string, shape: number[], values: number[] = []): Tensor {
  return {
    id,
    name: id,
    shape,
    values,
    dtype: "float64",
    axes: [],
    numel: product(shape),
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
function fixture(data = fixtures[0]) {
  const input = tensor("x", data.input.shape, data.input.values);
  const weight = tensor("w", data.weight.shape, data.weight.values);
  const output = tensor("y", data.output.shape, data.output.values);
  const bias = data.bias
    ? tensor("b", data.bias.shape, data.bias.values)
    : undefined;
  const op: Operation = {
    id: "conv",
    index: 0,
    kind: data.kind,
    function: `torch.${data.kind}`,
    inputs: bias ? ["x", "w", "b"] : ["x", "w"],
    outputs: ["y"],
    arguments: { ...data.arguments },
    source: null,
    module: "",
    error: null,
    status: "ok",
    lesson: {
      title: "",
      summary: "",
      detail: "",
      category: "compute",
      interaction: "convolution",
      mapping: null,
      axis_order: null,
    },
  };
  const run = {
    trace: {
      tensors: Object.fromEntries(
        [input, weight, output, ...(bias ? [bias] : [])].map((t) => [t.id, t]),
      ),
      operations: [op],
    },
  } as Run;
  return {
    run,
    op,
    input,
    weight,
    output,
    bias,
    p: tensorConvolution(run, op)!,
  };
}

describe("convolution coordinates and contributions", () => {
  it.each(fixtures)(
    "reconstructs every recorded output and both selection directions: $name",
    (data) => {
      const { p } = fixture(data);
      expect(p).not.toBeNull();
      for (let output = 0; output < p.output.numel; output++) {
        const terms = Array.from({ length: p.terms }, (_, term) =>
          convolutionSelection(p, output, term),
        );
        const sum = contributionSum(
          terms.map((s) =>
            finiteProduct(
              s.input === null ? 0 : p.input.values[s.input],
              p.weight.values[s.weight],
            ),
          ),
        )!;
        const bias = p.bias ? (p.bias.values[terms[0].feature] as number) : 0;
        expect(sum + bias).toBe(p.output.values[output]);
        for (const selected of terms) {
          expect(
            convolutionWeightSelection(p, selected.weight, output),
          ).toEqual({ output, term: selected.term });
          if (selected.input !== null) {
            const back = convolutionInputSelection(p, selected.input, output)!;
            expect(back).toEqual({ output, term: selected.term });
            expect(convolutionSelection(p, back.output, back.term).input).toBe(
              selected.input,
            );
          }
        }
      }
    },
  );
  it("matches a brute-force inverse for overlap, unreachable cells, stride, and dilation", () => {
    for (const stride of [1, 2, 3, 4])
      for (const dilation of [1, 2, 3, 4])
        for (const kernel of [1, 2, 4])
          for (const pad of [0, 1, 3]) {
            const outputs =
              Math.floor(
                (11 + 2 * pad - dilation * (kernel - 1) - 1) / stride,
              ) + 1;
            if (outputs < 1) continue;
            for (let x = 0; x < 11; x++)
              for (const previous of [0, outputs - 1]) {
                const choices: { output: number; kernel: number }[] = [];
                for (let o = 0; o < outputs; o++)
                  for (let k = 0; k < kernel; k++)
                    if (o * stride - pad + k * dilation === x)
                      choices.push({ output: o, kernel: k });
                const got = nearestConvolutionSample(
                  x,
                  pad,
                  stride,
                  dilation,
                  kernel,
                  outputs,
                  previous,
                );
                if (!choices.length) expect(got).toBeNull();
                else {
                  expect(choices).toContainEqual(got);
                  expect(Math.abs(got!.output - previous)).toBe(
                    Math.min(
                      ...choices.map((c) => Math.abs(c.output - previous)),
                    ),
                  );
                }
              }
          }
  });
  it("preserves batch and output-channel offset while crossing channel groups", () => {
    const { p } = fixture(fixtures[0]);
    const input = ravel([1, 3, 5], p.input.shape);
    const current = ravel([0, 1, 0], p.output.shape);
    const next = convolutionInputSelection(p, input, current)!;
    const selected = convolutionSelection(p, next.output, next.term);
    expect(selected.input).toBe(input);
    expect(selected.coordinates[0]).toBe(1);
    expect(selected.feature).toBe(4);
    expect(selected.group).toBe(1);
  });
  it("represents virtual padding without fabricating an input index", () => {
    const { p } = fixture(fixtures[3]);
    const selected = convolutionSelection(p, 0, 0);
    expect(selected.padding).toBe(true);
    expect(selected.input).toBeNull();
    expect(selected.inputCoordinates).toEqual([0, 0, -1, -2]);
    const same = fixture(fixtures[4]).p;
    expect(same.before).toEqual([0, 1]);
    expect(same.after).toEqual([1, 2]);
  });
  it("does not claim that stride-skipped input cells influence an output", () => {
    const { p } = fixture(fixtures[0]);
    expect(
      convolutionInputSelection(p, ravel([0, 0, 0], p.input.shape), 0),
    ).toBeNull();
  });
  it("keeps sampling bounded for billions of cells and millions of kernel terms", () => {
    const { run, op, input, weight, output } = fixture();
    input.shape = [1024, 16, 1048576];
    input.numel = product(input.shape);
    weight.shape = [32, 16, 100001];
    weight.numel = product(weight.shape);
    output.shape = [1024, 32, 1048576];
    output.numel = product(output.shape);
    op.arguments = { stride: 1, padding: 50000, dilation: 1, groups: 1 };
    op.inputs = ["x", "w"];
    const p = tensorConvolution(run, op)!;
    const cell = ravel([1023, 31, 1048575], p.output.shape);
    const window = convolutionWindow(p, cell, p.terms - 1);
    expect(window).toHaveLength(8);
    expect(window.at(-1)!.term).toBe(p.terms - 1);
    const end = convolutionInputSelection(p, p.input.numel - 1, cell)!;
    expect(convolutionSelection(p, end.output, end.term).input).toBe(
      p.input.numel - 1,
    );
    expect(convolutionSelection(p, end.output, end.term).coordinates[0]).toBe(
      1023,
    );
  });
  it("uses logical coordinates independently of recorded storage strides", () => {
    const { run, op, input } = fixture(fixtures[3]);
    input.contiguous = false;
    input.strides = [140, 1, 28, 4];
    input.storage_offset = 37;
    const p = tensorConvolution(run, op)!;
    const s = convolutionSelection(p, 12, 5);
    if (s.input !== null)
      expect(unravel(s.input, p.input.shape)).toEqual(s.inputCoordinates);
    expect(p.input.strides).toEqual([140, 1, 28, 4]);
  });
  it("reads older positional convolution arguments without replacing explicit invalid values", () => {
    const { run, op } = fixture();
    op.arguments = { arg3: 2, arg4: 1, arg5: 2, arg6: 2 };
    expect(tensorConvolution(run, op)?.groups).toBe(2);
    op.arguments.padding = null;
    expect(tensorConvolution(run, op)).toBeNull();
  });
  it("rejects failed, mutated, missing, unsafe, or incompatible geometry", () => {
    for (const patch of [
      { stride: 0 },
      { stride: [1, 2] },
      { padding: -1 },
      { groups: 3 },
      { groups: true },
      { dilation: 0 },
      { padding: "circular" },
      { stride: 2, padding: "same" },
      { dilation: Number.MAX_SAFE_INTEGER },
    ]) {
      const { run, op } = fixture();
      op.arguments = { ...op.arguments, ...patch };
      expect(tensorConvolution(run, op)).toBeNull();
    }
    const { run, op, weight } = fixture();
    op.status = "error";
    expect(tensorConvolution(run, op)).toBeNull();
    op.status = "ok";
    op.mutations = [{ before: "x", after: "y", kind: "write" }];
    expect(tensorConvolution(run, op)).toBeNull();
    op.mutations = [];
    weight.dtype = "int64";
    expect(tensorConvolution(run, op)).toBeNull();
    weight.dtype = "float64";
    op.inputs.push("missing");
    expect(tensorConvolution(run, op)).toBeNull();
  });
});

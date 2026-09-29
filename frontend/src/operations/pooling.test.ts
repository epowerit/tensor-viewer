import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import fixtures from "./fixtures/pooling.json";
import { product, ravel, unravel } from "../tensors/coordinates";
import {
  tensorPooling,
  poolingRegion,
  poolingSample,
  poolingSamples,
  poolingInputSelection,
  poolingCalculation,
  poolingWinner,
} from "./pooling";
function tensor(
  id: string,
  shape: number[],
  values: number[] = [],
  dtype = "float64",
): Tensor {
  return {
    id,
    name: id,
    shape,
    values,
    dtype,
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
  const outputs = data.outputs.map((t, i) =>
    tensor(`y${i}`, t.shape, t.values, i ? "int64" : "float64"),
  );
  const op: Operation = {
    id: "pool",
    index: 0,
    kind: data.kind,
    function: `torch.${data.kind}`,
    inputs: [input.id],
    outputs: outputs.map((t) => t.id),
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
      interaction: "pooling",
      mapping: null,
      axis_order: null,
    },
  };
  const run = {
    trace: {
      tensors: Object.fromEntries([input, ...outputs].map((t) => [t.id, t])),
      operations: [op],
    },
  } as Run;
  return { op, run, input, outputs, p: tensorPooling(run, op)! };
}
describe("exact pooling neighborhoods", () => {
  it.each(fixtures)(
    "matches every PyTorch output and inverse selection: $name",
    (data) => {
      const { p } = fixture(data);
      expect(p).not.toBeNull();
      for (let output = 0; output < p.output.numel; output++) {
        const region = poolingRegion(p, output),
          samples = poolingSamples(p, region, 0, true);
        const result = poolingCalculation(
          p,
          region,
          samples,
          (i) => p.input.values[i],
          p.output.values[output],
        );
        expect(result.complete).toBe(true);
        expect(result.result).toBeCloseTo(
          p.output.values[output] as number,
          12,
        );
        expect(samples.filter((s) => !s.padding)).toHaveLength(region.real);
        for (const sample of samples)
          if (sample.input !== null)
            expect(poolingInputSelection(p, sample.input, output)).toEqual({
              output,
              term: sample.term,
            });
        if (p.indices) {
          const winner = poolingWinner(p, region, p.indices.values[output]);
          expect(winner?.input).not.toBeNull();
          expect(p.input.values[winner!.input!]).toBe(p.output.values[output]);
        }
      }
    },
  );
  it("excludes ceil overhang from an average divisor but counts declared padding", () => {
    const included = fixture(fixtures[2]).p,
      excluded = fixture(fixtures[3]).p;
    const last = ravel([0, 0, 2, 2], included.output.shape);
    expect(poolingRegion(included, last)).toMatchObject({
      start: [3, 3],
      shape: [2, 2],
      terms: 4,
      real: 1,
      padded: 3,
      divisor: 4,
    });
    expect(poolingRegion(excluded, last).divisor).toBe(1);
    expect(poolingRegion(included, 0).divisor).toBe(9);
    expect(poolingRegion(fixture(fixtures[4]).p, 0).divisor).toBe(7);
    expect(poolingRegion(fixture(fixtures[5]).p, 0).divisor).toBe(-2);
  });
  it("keeps adaptive overlap and changing region sizes exact", () => {
    const p = fixture(fixtures[6]).p;
    expect(
      [0, 1, 2].map((r) => {
        const region = poolingRegion(p, ravel([0, 0, r, 0], p.output.shape));
        return [region.start[0], region.start[0] + region.shape[0]];
      }),
    ).toEqual([
      [0, 2],
      [1, 4],
      [3, 5],
    ]);
    for (const f of fixtures.filter((f) => f.kind === "adaptive_avg_pool2d")) {
      const p = fixture(f).p;
      for (let input = 0; input < p.input.numel; input++) {
        const selected = poolingInputSelection(p, input, p.output.numel - 1)!;
        expect(
          poolingSample(p, poolingRegion(p, selected.output), selected.term)
            .input,
        ).toBe(input);
      }
    }
  });
  it("distinguishes a recorded max index from ties and never uses a global batch index", () => {
    const { p } = fixture(fixtures[1]);
    p.input.values = Array(p.input.numel).fill(-2);
    const output = ravel([1, 1, 1, 1], p.output.shape),
      region = poolingRegion(p, output);
    const samples = poolingSamples(p, region, 0, true);
    const calculation = poolingCalculation(p, region, samples, () => -2, -2);
    expect(calculation.matches).toHaveLength(region.real);
    const spatialIndex = ravel([1, 1], p.input.shape.slice(-2));
    const selected = poolingWinner(p, region, spatialIndex)!;
    expect(selected.coordinates).toEqual([1, 1, 1, 1]);
    expect(selected.input).toBe(ravel([1, 1, 1, 1], p.input.shape));
    expect(poolingWinner(p, region, p.input.numel - 1)).toBeNull();
    expect(poolingWinner(p, region, 0)).toBeNull(); // dilation skips this coordinate
    expect(poolingWinner(p, region, "nan")).toBeNull();
  });
  it("matches exhaustive inverse membership, including skipped stride/dilation cells", () => {
    const { run, op, input, outputs } = fixture(fixtures[1]);
    for (const kernel of [1, 2, 3])
      for (const stride of [1, 2, 3])
        for (const dilation of [1, 2]) {
          input.shape = [1, 5, 6];
          input.numel = 30;
          const height =
            Math.floor((5 - dilation * (kernel - 1) - 1) / stride) + 1;
          const width =
            Math.floor((6 - dilation * (kernel - 1) - 1) / stride) + 1;
          outputs.forEach((t) => {
            t.shape = [1, height, width];
            t.numel = height * width;
          });
          op.arguments = { kernel_size: kernel, stride, dilation };
          const p = tensorPooling(run, op)!;
          expect(p).not.toBeNull();
          const sets = Array.from({ length: p.output.numel }, (_, o) =>
            poolingSamples(p, poolingRegion(p, o), 0, true),
          );
          for (let x = 0; x < input.numel; x++)
            for (const previous of [0, p.output.numel - 1]) {
              const possible = sets.flatMap((s, o) =>
                s.some((cell) => cell.input === x) ? [o] : [],
              );
              const got = poolingInputSelection(p, x, previous);
              if (!possible.length) expect(got).toBeNull();
              else {
                expect(possible).toContain(got!.output);
                expect(sets[got!.output][got!.term].input).toBe(x);
                if (possible.includes(previous))
                  expect(got!.output).toBe(previous);
              }
            }
        }
  });
  it("keeps huge adaptive windows bounded and ratio arithmetic exact", () => {
    const { run, op, input, outputs } = fixture(fixtures[6]);
    input.shape = [1024, 4, 1024, 1024];
    input.numel = product(input.shape);
    outputs[0].shape = [1024, 4, 1, 1];
    outputs[0].numel = product(outputs[0].shape);
    op.arguments = { output_size: [1, 1] };
    let p = tensorPooling(run, op)!;
    const region = poolingRegion(p, p.output.numel - 1);
    expect(region.terms).toBe(1048576);
    const samples = poolingSamples(p, region, region.terms - 1, true);
    expect(samples).toHaveLength(8);
    expect(samples.at(-1)!.input).toBe(p.input.numel - 1);
    expect(poolingCalculation(p, region, samples, () => 1, 1)).toMatchObject({
      complete: false,
      aggregate: 8,
      result: undefined,
      matches: [],
    });
    input.shape = [1, 1099511627775, 1];
    input.numel = product(input.shape);
    outputs[0].shape = [1, 1099511627774, 1];
    outputs[0].numel = product(outputs[0].shape);
    op.arguments = { output_size: [1099511627774, 1] };
    p = tensorPooling(run, op)!;
    expect(poolingRegion(p, p.output.numel - 1)).toMatchObject({
      start: [1099511627773, 0],
      shape: [2, 1],
    });
    const selected = poolingInputSelection(p, p.input.numel - 1, 0)!;
    expect(
      poolingSample(p, poolingRegion(p, selected.output), selected.term).input,
    ).toBe(p.input.numel - 1);
  });
  it("does not turn missing/nonfinite data or partial maxima into a winner", () => {
    const { p } = fixture(fixtures[0]);
    const region = poolingRegion(p, 0),
      samples = poolingSamples(p, region, 0, true);
    for (const value of [undefined, "nan", "-inf", Infinity])
      expect(
        poolingCalculation(p, region, samples, () => value, -1),
      ).toMatchObject({ aggregate: undefined, result: undefined, matches: [] });
    expect(
      poolingCalculation(p, region, samples.slice(0, 2), () => -1, -1).matches,
    ).toEqual([]);
    const average = fixture(fixtures[2]).p,
      r = poolingRegion(average, 0);
    expect(
      poolingCalculation(
        average,
        r,
        poolingSamples(average, r, 0, true),
        () => undefined,
        0,
      ).aggregate,
    ).toBeUndefined();
  });
  it("reads old positional arguments and validates recorded output geometry", () => {
    const { run, op } = fixture(fixtures[2]);
    op.arguments = { arg1: 3, arg2: 2, arg3: 1, arg4: true, arg5: true };
    expect(tensorPooling(run, op)?.ceil).toBe(true);
    for (const patch of [
      { stride: 0 },
      { padding: 3 },
      { ceil_mode: 1 },
      { divisor_override: 0 },
      { count_include_pad: null },
      { kernel_size: [2, 3, 4] },
    ]) {
      const f = fixture(fixtures[2]);
      f.op.arguments = { ...f.op.arguments, ...patch };
      expect(tensorPooling(f.run, f.op)).toBeNull();
    }
    const f = fixture(fixtures[1]);
    f.outputs[1].dtype = "float64";
    expect(tensorPooling(f.run, f.op)).toBeNull();
    f.outputs[1].dtype = "int64";
    f.op.status = "error";
    expect(tensorPooling(f.run, f.op)).toBeNull();
    f.op.status = "ok";
    f.op.mutations = [{ before: "x", after: "y0", kind: "write" }];
    expect(tensorPooling(f.run, f.op)).toBeNull();
  });
  it("maps non-contiguous input in logical coordinate order", () => {
    const { p } = fixture(fixtures[0]);
    p.input.contiguous = false;
    p.input.strides = [105, 35, 1, 5];
    p.input.storage_offset = 19;
    const selected = poolingInputSelection(p, 42, 0)!;
    const sample = poolingSample(
      p,
      poolingRegion(p, selected.output),
      selected.term,
    );
    expect(sample.coordinates).toEqual(unravel(42, p.input.shape));
    expect(sample.input).toBe(42);
  });
});

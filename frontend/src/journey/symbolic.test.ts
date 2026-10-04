import { expect, test } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import { axisSymbol, symbolicShapes } from "./symbolic";

type Trace = Run["trace"];

/** A small model's trace at batch b and tokens t. */
function trace(b: number, t: number): Trace {
  const shapes: [string, number[]][] = [
    ["linear", [b, t, 48]],
    ["reshape", [b, t, 3, 16]],
    ["matmul", [b, t, t]],
    ["flatten", [b * t, 48]],
    ["conv", [b, Math.floor(t / 2)]],
    ["pad", [b, t + 2]],
  ];
  const tensors: Record<string, Tensor> = {};
  const operations = shapes.map(([kind, shape], i) => {
    tensors[`t${i}`] = { id: `t${i}`, shape, dtype: "float32" } as Tensor;
    return {
      id: `op${i}`,
      kind,
      function: kind,
      module: "M",
      outputs: [`t${i}`],
      source: { line: i + 1, text: kind, file: null },
    } as unknown as Operation;
  });
  return { operations, tensors } as unknown as Trace;
}

test("result sizes read in the input's axes, from doubled-axis shape checks", () => {
  const base = trace(2, 6);
  const shapes = symbolicShapes(base, [
    { symbol: "B", size: 2, trace: trace(4, 6) },
    { symbol: "T", size: 6, trace: trace(2, 12) },
  ]);
  expect(shapes.get("t0")).toEqual(["B", "T", "48"]);
  expect(shapes.get("t1")).toEqual(["B", "T", "3", "16"]);
  expect(shapes.get("t2")).toEqual(["B", "T", "T"]);
  expect(shapes.get("t3")).toEqual(["B·T", "48"]);
  expect(shapes.get("t4")).toEqual(["B", "T/2"]);
  expect(shapes.get("t5")).toEqual(["B", "T+2"]);
});

test("axis names become the usual symbols", () => {
  expect(
    ["batch", "tokens", "channels", "height", "width", "features", null].map(
      (name, axis) => axisSymbol(name, axis),
    ),
  ).toEqual(["B", "T", "C", "H", "W", "D", "N6"]);
});

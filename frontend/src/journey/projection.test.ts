import { expect, test } from "vitest";
import type { Run, Tensor } from "../api/client";
import { evaluateSize, project, recordedSymbols } from "./projection";

test("symbolic sizes evaluate at other symbol values", () => {
  const at = { B: 4, T: 10 };
  expect(evaluateSize("48", at)).toBe(48);
  expect(evaluateSize("B·T", at)).toBe(40);
  expect(evaluateSize("3·T", at)).toBe(30);
  expect(evaluateSize("T/2", at)).toBe(5);
  expect(evaluateSize("T+2", at)).toBe(12);
  expect(evaluateSize("2·T−2", at)).toBe(18);
  expect(evaluateSize("T²", at)).toBe(100);
  expect(evaluateSize("2·B·T", at)).toBe(80);
  expect(evaluateSize("C", at)).toBeNull();
  expect(evaluateSize("T)(", at)).toBeNull();
});

test("a run projected at a larger size: compute and peak memory follow the shapes", () => {
  // x [B, T, 8] → linear to [B, T, 16] with a [16, 8] weight.
  const tensor = (id: string, shape: number[], role = "intermediate") =>
    ({
      id,
      name: id,
      shape,
      numel: shape.reduce((a, b) => a * b, 1),
      dtype: "float32",
      storage_id: id,
      role,
    }) as unknown as Tensor;
  const trace = {
    input_ids: ["x"],
    output_ids: ["y"],
    tensors: {
      x: tensor("x", [1, 2, 8], "input"),
      w: tensor("w", [16, 8], "parameter"),
      y: tensor("y", [1, 2, 16]),
    },
    operations: [
      { id: "op0", kind: "linear", inputs: ["x", "w"], outputs: ["y"] },
    ],
  } as unknown as Run["trace"];
  const labels: Record<string, string[]> = {
    x: ["B", "T", "8"],
    y: ["B", "T", "16"],
  };
  const labelsOf = (id: string) => labels[id] ?? null;
  expect(recordedSymbols(trace.tensors.x, labels.x)).toEqual({ B: 1, T: 2 });
  const recorded = project(trace, labelsOf, { B: 1, T: 2 });
  // 2·M·N·K: 2 · (1·2·16) · 8 = 512 FLOPs; x and y alive: 64 + 128 bytes.
  expect(recorded).toEqual({ flops: 512, peakBytes: 192, kept: 0 });
  const larger = project(trace, labelsOf, { B: 32, T: 1024 });
  expect(larger.flops).toBe(512 * 32 * 512);
  expect(larger.peakBytes).toBe(192 * 32 * 512);
});

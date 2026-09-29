import { describe, expect, it } from "vitest";
import type { Operation, Tensor } from "../api/client";
import { sourceIndex, outputIndices } from "./relationships";
const tensor = (shape: number[]) => ({ shape }) as Tensor;
const operation = (
  rule: "identity" | "permutation" | "unfold" | "roll",
  axis_order: number[] | null = null,
  args: Record<string, unknown> = {},
) =>
  ({
    lesson: { mapping_rule: rule, axis_order, mapping: null },
    arguments: args,
  }) as unknown as Operation;
describe("bounded tensor relationships", () => {
  it("maps billion-element permutations without allocating a full index map", () => {
    const input = tensor([1024, 512, 1024]),
      output = tensor([1024, 1024, 512]);
    const op = operation("permutation", [0, 2, 1]);
    const target = 1023 * 1024 * 512 + 1000 * 512 + 500;
    const source = 1023 * 512 * 1024 + 500 * 1024 + 1000;
    expect(sourceIndex(op, input, output, target)).toBe(source);
    expect(outputIndices(op, input, output, source)).toEqual([target]);
  });
  it("keeps identity relationships exact across a reshape", () => {
    expect(
      sourceIndex(
        operation("identity"),
        tensor([1024, 1024, 1024]),
        tensor([1024, 1048576]),
        1073741823,
      ),
    ).toBe(1073741823);
  });
  it("finds overlapping windows and caps exceptionally large inverse memberships", () => {
    const op = operation("unfold", null, { dimension: 0, size: 3, step: 1 });
    expect(outputIndices(op, tensor([5]), tensor([3, 3]), 2)).toEqual([
      2, 4, 6,
    ]);
    expect(
      [2, 4, 6].map((index) =>
        sourceIndex(op, tensor([5]), tensor([3, 3]), index),
      ),
    ).toEqual([2, 2, 2]);
    const huge = operation("unfold", null, {
      dimension: 0,
      size: 100000,
      step: 1,
    });
    expect(
      outputIndices(huge, tensor([200000]), tensor([100001, 100000]), 100000),
    ).toHaveLength(256);
  });
});

describe("cyclic roll mappings", () => {
  it("reverses multidimensional shifts, repeated dimensions and flattening", () => {
    for (const args of [
      { shifts: [1, -2], dims: [1, 2] },
      { shifts: [5, -1], dims: [1, 1] },
      { shifts: 7 },
      { shifts: -99, dims: -1 },
    ]) {
      const op = operation("roll", null, args),
        t = tensor([2, 3, 4]);
      for (let i = 0; i < 24; i++) {
        const j = outputIndices(op, t, t, i)[0];
        expect(sourceIndex(op, t, t, j)).toBe(i);
        expect(j).toBeGreaterThanOrEqual(0);
        expect(j).toBeLessThan(24);
      }
    }
    const op = operation("roll", null, { shifts: [-1, -1], dims: [2, 3] }),
      t = tensor([1024, 768, 128, 128]);
    expect(sourceIndex(op, t, t, 0)).toBe(129);
  });
});

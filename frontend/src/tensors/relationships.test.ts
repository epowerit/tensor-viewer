import { describe, expect, it } from "vitest";
import type { Operation, Tensor } from "../api/client";
import { sourceIndex, outputIndices } from "./relationships";
const tensor = (shape: number[]) => ({ shape }) as Tensor;
const operation = (
  rule: "identity" | "permutation" | "unfold",
  axis_order: number[] | null = null,
  args: Record<string, number> = {},
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

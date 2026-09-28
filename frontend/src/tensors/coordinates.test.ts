import { describe, expect, it } from "vitest";
import {
  dotContributors,
  normalizationGroup,
  ravel,
  unravel,
} from "./coordinates";

describe("tensor coordinate relationships", () => {
  it("converts every element of a 6D tensor without mixing axes", () => {
    const shape = [2, 2, 3, 2, 2, 4];
    for (let i = 0; i < 192; i++)
      expect(ravel(unravel(i, shape), shape)).toBe(i);
    expect(unravel(25, [1, 2, 3, 5])).toEqual([0, 1, 2, 0]);
    expect(unravel(0, [])).toEqual([]);
  });
  it("links exact operand cells for broadcasting matrix multiplication", () => {
    const pairs = dotContributors(
      [2, 1, 3, 4],
      [5, 4, 2],
      [2, 5, 3, 2],
      ravel([1, 4, 2, 1], [2, 5, 3, 2]),
    );
    expect(pairs).toEqual([
      { left: 20, right: 33 },
      { left: 21, right: 35 },
      { left: 22, right: 37 },
      { left: 23, right: 39 },
    ]);
  });
  it("selects normalization groups across a non-last axis", () => {
    expect(normalizationGroup([2, 3, 4], 17, 1)).toEqual([13, 17, 21]);
    expect(normalizationGroup([2, 3, 4], 17, -1)).toEqual([16, 17, 18, 19]);
  });
});

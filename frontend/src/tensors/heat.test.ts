import { expect, test } from "vitest";
import { heatFill, heatGradient, heatLevel, heatRangeOf } from "./heat";

test("a range across zero shades diverging around zero", () => {
  expect(heatLevel(0, -2, 4)).toBe(0);
  expect(heatLevel(4, -2, 4)).toBe(1);
  expect(heatLevel(-2, -2, 4)).toBe(-0.5);
  expect(heatLevel(9, -2, 4)).toBe(1);
});

test("a one-sided range shades from its smallest magnitude to its largest", () => {
  expect(heatLevel(0.2, 0.2, 1)).toBe(0);
  expect(heatLevel(1, 0.2, 1)).toBe(1);
  expect(heatLevel(-1, -1, -0.5)).toBe(-1);
  expect(heatLevel(-0.5, -1, -0.5)).toBe(-0);
  expect(heatLevel(3, 3, 3)).toBe(1);
  expect(heatLevel(Number.NaN, 0, 1)).toBeNull();
});

test("fills deepen with the level and the legend matches them", () => {
  expect(heatFill(0)).toContain(" 8%");
  expect(heatFill(-1)).toContain("#4b93e0 80%");
  expect(heatFill(1)).toContain("#dc5c7e 80%");
  expect(heatGradient(-1, 3)).toContain("var(--tensor-base, #221d2d) 25%");
});

test("heat spans the whole tensor's finite range, when it has one", () => {
  expect(
    heatRangeOf({ minimum: -1, maximum: 2, value_source: "inline" }),
  ).toEqual({ low: -1, high: 2 });
  expect(
    heatRangeOf({ minimum: -1, maximum: 2, value_source: "shape" }),
  ).toBeNull();
  expect(
    heatRangeOf({ minimum: null, maximum: 2, value_source: "paged" }),
  ).toBeNull();
});

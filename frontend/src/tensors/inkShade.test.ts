import { expect, test } from "vitest";
import { hexToHsl, inkShade, storyAxis } from "./inkShade";

const ink = (...colors: string[]) => ({
  colors,
  piece: null,
  text: "x.tokens",
});

test("hex colors convert to HSL", () => {
  expect(hexToHsl("#ff0000")).toEqual([0, 100, 50]);
  const [h, , l] = hexToHsl("#6fd3bf");
  expect(Math.round(h)).toBe(168);
  expect(Math.round(l)).toBe(63);
});

test("an axis ramps from dark to light in its own hue", () => {
  const first = inkShade(ink("#f2b27c"), 0, 4);
  const last = inkShade(ink("#f2b27c"), 3, 4);
  expect(first.fill).toMatch(/^hsl\(27 \d+% 30%\)$/);
  expect(last.fill).toMatch(/^hsl\(27 \d+% 66%\)$/);
  // Labels stay readable on both ends.
  expect(first.text).toBe("#ffffff");
  expect(last.text).toBe("#1b1525");
  // A single position sits in the middle of the ramp.
  expect(inkShade(ink("#f2b27c"), 0, 1).fill).toMatch(/48%\)$/);
});

test("a merged axis walks through each source's ink in order", () => {
  const merged = ink("#6fd3bf", "#f2b27c");
  expect(inkShade(merged, 0, 8).fill).toMatch(/^hsl\(168 /);
  expect(inkShade(merged, 7, 8).fill).toMatch(/^hsl\(27 /);
});

test("the default color axis is the one the step moves or changes", () => {
  const batch = ink("#6fd3bf"),
    tokens = { ...ink("#f2b27c"), text: "x.tokens" },
    features = { ...ink("#b69cf8"), text: "x.features" };
  batch.text = "x.batch";
  // transpose(1, 2) of [batch, tokens, features]: tokens and features move.
  expect(
    storyAxis([2, 4, 8], [batch, tokens, features], [batch, features, tokens]),
  ).toBe(2);
  // reshape splitting features: features changes, the others stay.
  expect(
    storyAxis(
      [2, 4, 8],
      [batch, tokens, features],
      [batch, tokens, null, null],
    ),
  ).toBe(2);
  // Nothing moves or changes: the last inked axis.
  expect(storyAxis([2, 4], [batch, tokens], [batch, tokens])).toBe(1);
  expect(storyAxis([2, 4], null, null)).toBeNull();
});

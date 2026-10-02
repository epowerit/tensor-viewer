import { expect, test } from "vitest";
import { axisPhrase } from "./axisPhrase";

test("axes are named when every one has a real name", () => {
  expect(axisPhrase([1], ["tokens", "experts"])).toBe("axis 1 (experts)");
  expect(axisPhrase([1, 2], ["b", "heads", "tokens"])).toBe(
    "axes 1, 2 (heads, tokens)",
  );
  expect(axisPhrase([1], ["axis 0", "axis 1"])).toBe("axis 1");
  expect(axisPhrase([0])).toBe("axis 0");
});

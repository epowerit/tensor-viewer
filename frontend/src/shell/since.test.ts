import { expect, test } from "vitest";
import { since } from "./since";

const now = new Date(2026, 9, 2, 20, 30, 0);
const at = (...parts: number[]) =>
  since(new Date(2026, ...(parts as [number])), now);

test("recent runs read in minutes and hours, older ones by day", () => {
  expect(at(9, 2, 20, 29, 40)).toBe("just now");
  expect(at(9, 2, 20, 18, 0)).toBe("12 min ago");
  expect(at(9, 2, 17, 10, 0)).toBe("3 h ago");
  expect(at(9, 1, 23, 50, 0)).toBe("yesterday");
  expect(at(8, 28, 9, 0, 0)).toBe(
    new Date(2026, 8, 28).toLocaleDateString([], { weekday: "long" }),
  );
  expect(at(8, 1, 9, 0, 0)).toBe(
    new Date(2026, 8, 1).toLocaleDateString([], {
      month: "short",
      day: "numeric",
    }),
  );
  expect(since(new Date(2025, 0, 5), now)).toContain("2025");
  expect(since(new Date("nope"), now)).toBe("");
});

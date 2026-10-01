import { expect, test } from "vitest";
import type { LineResult } from "../console/script";
import { checkContracts, parseContract } from "./contracts";

const result = (shape: number[], extra: Partial<LineResult> = {}) =>
  ({
    operations: [],
    output: { name: "t", shape },
    error: null,
    fresh: true,
    ...extra,
  }) as unknown as LineResult;
const check = (lines: string[], shapes: (number[] | LineResult | null)[]) =>
  checkContracts(
    lines,
    new Map(
      shapes.flatMap((item, i) =>
        item === null
          ? []
          : [[i + 1, Array.isArray(item) ? result(item) : item] as const],
      ),
    ),
  );

test("contracts read sizes, names, wildcards, and a rest marker", () => {
  expect(parseContract("y = x.flatten(1)", 1)).toBeNull();
  expect(parseContract("y = f(x)  # shape: [B, 12]", 3)).toEqual({
    line: 3,
    text: "[B, 12]",
    entries: [
      { kind: "name", name: "B" },
      { kind: "size", size: 12 },
    ],
  });
  expect(parseContract("# shape: (batch tokens _)", 1)).toMatchObject({
    text: "[batch, tokens, _]",
  });
  expect(parseContract("# shape: ..., D", 1)).toMatchObject({
    entries: [{ kind: "rest" }, { kind: "name", name: "D" }],
  });
  expect(parseContract("# shape: 2.5", 1)).toEqual({
    line: 1,
    error: "“2.5” is not a size, a name, _, or ...",
  });
  expect(parseContract("# shape: ..., ...", 1)).toMatchObject({
    error: "Use ... at most once in a contract.",
  });
});

test("names bind on first use and must agree afterwards", () => {
  const lines = [
    "q = x.reshape(2, 4, 8)  # shape: B, T, D",
    "k = q.transpose(1, 2)  # shape: B, D, T",
    "s = q @ k  # shape: B, T, T",
    "bad = q.flatten(1)  # shape: B, D",
  ];
  const checks = check(lines, [
    [2, 4, 8],
    [2, 8, 4],
    [2, 4, 4],
    [2, 32],
  ]);
  expect(checks.map((c) => c.ok)).toEqual([true, true, true, false]);
  expect(checks[3].message).toBe(
    "D was 8 on line 1, but axis 1 of t is 32: [2, 32].",
  );
  // The same name twice in one contract must match itself.
  expect(check(["# shape: N, N"], [[3, 4]])[0].message).toContain(
    "names two axes of different sizes here (3 and 4)",
  );
});

test("exact sizes, rank, wildcards, and ... are enforced", () => {
  const [size, rank, rest, tooShort, wild] = [
    check(["# shape: 2, 13"], [[2, 12]])[0],
    check(["# shape: 2, 12"], [[2, 3, 4]])[0],
    check(["# shape: ..., 4"], [[2, 3, 4]])[0],
    check(["# shape: B, ..., 4, 5"], [[4, 5]])[0],
    check(["# shape: _, _"], [[7, 9]])[0],
  ];
  expect(size.message).toBe("Axis 1 of t should be 13, but it is 12: [2, 12].");
  expect(rank.message).toBe("Expected 2 axes [2, 12], but t has 3: [2, 3, 4].");
  expect(rest.ok).toBe(true);
  expect(tooShort.message).toContain("Expected at least 3 axes");
  expect(wild.ok).toBe(true);
});

test("lines without a current tensor are skipped rather than failed", () => {
  const lines = ["a = x  # shape: B, 3", "b = a  # shape: B, 3"];
  expect(check(lines, [null, [2, 3]]).map((c) => c.line)).toEqual([2]);
  expect(
    check(lines, [result([2, 3], { fresh: false }), result([2, 3])]),
  ).toHaveLength(1);
  expect(
    check(lines, [result([2, 3], { error: "RuntimeError: x" }), null]),
  ).toEqual([]);
  // Unreadable contracts are reported even without a tensor.
  expect(check(["a = x  # shape: 3.0"], [null])[0].ok).toBe(false);
});

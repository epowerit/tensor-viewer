import { expect, test } from "vitest";
import type { Operation, Tensor } from "../api/client";
import { presenters } from "./presenters";

const tensor = (
  name: string,
  shape: number[],
  values: number[],
  dtype = "float32",
) =>
  ({
    id: name,
    name,
    shape,
    values,
    dtype,
    numel: shape.reduce((a, b) => a * b, 1),
  }) as Tensor;
const operation = (kind: string, relation: object, args: object = {}) =>
  ({
    kind,
    status: "ok",
    arguments: args,
    lesson: { interaction: "relation", relation, mapping: null },
  }) as unknown as Operation;
const elementwise = (roles: string[], extra: object = {}) => ({
  rule: "elementwise",
  operand: 0,
  roles,
  ...extra,
});

test("a reduction shows its whole group and the recorded result", () => {
  const x = tensor("x", [2, 3], [1, 2, 3, 4, 5, 6]);
  const reduce = { rule: "reduce", operand: 0, axes: [1], keepdim: false };
  const sum = presenters.relation(
    operation("sum", reduce),
    [x],
    tensor("s", [2], [6, 15]),
    1,
  );
  expect(sum.operandHighlights).toEqual([[3, 4, 5]]);
  expect(sum.expression).toBe("4 + 5 + 6 = 15");
  const mean = presenters.relation(
    operation("mean", reduce),
    [x],
    tensor("m", [2], [2, 5]),
    0,
  );
  expect(mean.expression).toBe("(1 + 2 + 3) ÷ 3 = 2");
  const largest = presenters.relation(
    operation("argmax", reduce),
    [x],
    tensor("i", [2], [2, 2], "int64"),
    0,
  );
  expect(largest.expression).toBe("argmax(1, 2, 3) = 2");
  // Without recorded values the group is still shown, with no arithmetic.
  const shapes = presenters.relation(
    operation("sum", reduce),
    [{ ...x, values: [] }],
    tensor("s", [2], []),
    0,
  );
  expect(shapes.leftHighlights).toEqual([0, 1, 2]);
  expect(shapes.expression).toBeUndefined();
});

test("elementwise rules highlight every operand and name the arithmetic", () => {
  const x = tensor("x", [2, 2], [1, 2, 3, 4]),
    column = tensor("c", [2, 1], [10, 20]);
  const product = presenters.relation(
    operation("mul", elementwise(["input", "other"])),
    [x, column],
    tensor("y", [2, 2], [10, 20, 60, 80]),
    3,
  );
  expect(product.operandHighlights).toEqual([[3], [1]]);
  expect(product.expression).toBe("4 × 20 = 80");
  const reversed = presenters.relation(
    operation("__rsub__", elementwise(["input"], { reversed: true }), {
      other: 10,
    }),
    [x],
    tensor("y", [2, 2], [9, 8, 7, 6]),
    0,
  );
  expect(reversed.expression).toBe("10 − 1 = 9");
  const unary = presenters.relation(
    operation("relu", elementwise(["input"])),
    [tensor("x", [2], [-3, 2])],
    tensor("y", [2], [0, 2]),
    0,
  );
  expect(unary.expression).toBe("relu(-3) = 0");
  const compared = presenters.relation(
    operation("gt", elementwise(["input"]), { other: 2 }),
    [x],
    tensor("m", [2, 2], [0, 0, 1, 1], "bool"),
    2,
  );
  expect(compared.expression).toBe("3 > 2 = True");
});

test("arithmetic equations retain alpha, reverse order, and division rounding", () => {
  const explain = (
    kind: string,
    args: object,
    input: number,
    other: number,
    result: number,
  ) =>
    presenters.relation(
      operation(kind, elementwise(["input", "other"]), args),
      [tensor("x", [1], [input]), tensor("y", [1], [other])],
      tensor("out", [1], [result]),
      0,
    ).expression;
  expect(explain("add", { alpha: 2 }, 5, 2, 9)).toBe("5 + 2 × 2 = 9");
  expect(explain("sub", { alpha: 2 }, 5, 2, 1)).toBe("5 − 2 × 2 = 1");
  expect(explain("rsub", { alpha: 2 }, 5, 2, -8)).toBe("2 − 2 × 5 = -8");
  expect(explain("rsub", {}, 5, 2, -3)).toBe("2 − 5 = -3");
  expect(explain("div", { rounding_mode: "floor" }, -5, 2, -3)).toBe(
    "floor(-5 ÷ 2) = -3",
  );
  expect(explain("div", { rounding_mode: "trunc" }, -5, 2, -2)).toBe(
    "trunc(-5 ÷ 2) = -2",
  );
  expect(explain("div", {}, 5, 2, 2.5)).toBe("5 ÷ 2 = 2.5");
  expect(explain("__and__", {}, 1, 2, 0)).toBe("1 & 2 = 0");
  expect(explain("__or__", {}, 1, 2, 3)).toBe("1 | 2 = 3");
});

test("parameterized unary calls keep their recorded settings", () => {
  const explain = (kind: string, args: object, input: number, result: number) =>
    presenters.relation(
      operation(kind, elementwise(["input"]), args),
      [tensor("x", [1], [input])],
      tensor("out", [1], [result]),
      0,
    ).expression;
  expect(explain("round", { decimals: 2 }, 1.234, 1.23)).toBe(
    "round(1.234, decimals=2) = 1.23",
  );
  expect(
    explain("dropout", { p: 0.25, training: true, inplace: false }, 3, 0),
  ).toBe("dropout(3, p=0.25, training=True, inplace=False) = 0");
  expect(explain("clamp", { min: 0, max: 1 }, -2, 0)).toBe(
    "clamp(-2, min=0, max=1) = 0",
  );
  expect(explain("leaky_relu", { negative_slope: 0.1 }, -2, -0.2)).toBe(
    "leaky_relu(-2, negative_slope=0.1) = -0.2",
  );
});

test("masks and triangles explain which branch a cell took", () => {
  const x = tensor("x", [2, 2], [1, 2, 3, 4]),
    mask = tensor("m", [2, 2], [0, 1, 0, 1], "bool");
  const where = operation("where", elementwise(["condition", "input"]), {
    other: 0,
  });
  const chosen = tensor("y", [2, 2], [0, 2, 0, 4]);
  expect(presenters.relation(where, [mask, x], chosen, 1).expression).toBe(
    "condition is True → first value 2 = 2",
  );
  expect(presenters.relation(where, [mask, x], chosen, 2).expression).toBe(
    "condition is False → second value 0 = 0",
  );
  const filled = presenters.relation(
    operation("masked_fill", elementwise(["input", "mask"]), { value: -1 }),
    [x, mask],
    tensor("y", [2, 2], [1, -1, 3, -1]),
    3,
  );
  expect(filled.expression).toBe("mask is True → filled with -1 = -1");
  const triangle = operation("tril", elementwise(["input"], { diagonal: 0 }));
  const lower = tensor("y", [2, 2], [1, 0, 3, 4]);
  expect(presenters.relation(triangle, [x], lower, 1).title).toBe(
    "Outside the triangle: zero",
  );
  expect(presenters.relation(triangle, [x], lower, 2).title).toBe(
    "Inside the triangle: kept",
  );
});

test("selection rules name the source cell and fall back when unverified", () => {
  const x = tensor("x", [2, 3], [1, 2, 3, 4, 5, 6]);
  const pick = operation("__getitem__", {
    rule: "index",
    operand: 0,
    axes: [
      { kind: "int", index: 1 },
      { kind: "slice", start: 1, step: 1, out: 0 },
    ],
  });
  const row = tensor("row", [2], [5, 6]);
  const result = presenters.relation(pick, [x], row, 1);
  expect(result.leftHighlights).toEqual([5]);
  expect(result.expression).toBe("x[1, 2] → [1] = 6");
  const wrong = presenters.relation(
    pick,
    [x],
    tensor("row", [3], [4, 5, 6]),
    0,
  );
  expect(wrong.title).toBe("Inspect an element");
});

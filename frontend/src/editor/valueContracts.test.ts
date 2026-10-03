import { expect, test } from "vitest";
import type { LineResult } from "../console/script";
import { checkContracts } from "./contracts";
import {
  acceptRecorded,
  checkValueContracts,
  sumsKey,
  sumsQuestions,
  contractFor,
  mergeChecks,
  parseValueContract,
} from "./valueContracts";

const result = (
  output: Record<string, unknown>,
  extra: Partial<LineResult> = {},
) =>
  ({
    operations: [],
    error: null,
    fresh: true,
    output: {
      name: "probs",
      shape: [2, 3],
      numel: 6,
      value_source: "inline",
      values: [0.2, 0.3, 0.5, 0.1, 0.1, 0.8],
      minimum: 0.1,
      maximum: 0.8,
      histogram: { mean: 1 / 3, non_finite: 0 },
      ...output,
    },
    ...extra,
  }) as unknown as LineResult;

test("value contracts read their clauses and leave ordinary comments alone", () => {
  expect(
    parseValueContract("p = s.softmax(-1)  # range: 0..1; sums(-1): 1", 4),
  ).toEqual({
    line: 4,
    text: "range: 0..1; sums(-1): 1",
    clauses: [
      { kind: "range", low: 0, high: 1 },
      { kind: "sums", axis: -1, value: 1 },
    ],
  });
  expect(
    parseValueContract("h = f(x)  # shape: B, T; finite; mean: 0 ± 0.5", 1),
  ).toMatchObject({
    clauses: [{ kind: "finite" }, { kind: "mean", value: 0, tolerance: 0.5 }],
  });
  expect(parseValueContract("y = x  # range: ..1e-3", 1)).toMatchObject({
    clauses: [{ kind: "range", low: null, high: 1e-3 }],
  });
  for (const comment of [
    "# mean pooling over tokens",
    "# finite differences",
    "# shape: B, T",
  ])
    expect(parseValueContract(`y = x  ${comment}`, 1)).toBeNull();
  expect(parseValueContract("y = x  # range: low..high", 1)).toHaveProperty(
    "error",
  );
});

test("value contracts are checked against the recorded values", () => {
  const lines = [
    "probs = s.softmax(-1)  # range: 0..1; sums(-1): 1; mean: 0.33 ± 0.01",
  ];
  expect(
    checkValueContracts(lines, new Map([[1, result({})]]))[0],
  ).toMatchObject({ ok: true });
  const broken = checkValueContracts(
    lines,
    new Map([
      [
        1,
        result({
          values: [0.2, 0.3, 0.6, 0.1, 0.1, 0.8],
          maximum: 1.2,
          histogram: { mean: 0.35, non_finite: 0 },
        }),
      ],
    ]),
  )[0];
  expect(broken.ok).toBe(false);
  expect(broken.message).toContain("reaches 1.2, above 1.");
  expect(broken.message).toContain(
    "1 of 2 sums over axis -1 are not 1; one is 1.1.",
  );
  expect(broken.message).toContain("mean is 0.35, not 0.33 ± 0.01.");
  // NaN makes the mean not a number, as in torch.
  const faulty = checkValueContracts(
    ["h = f(x)  # finite; mean: 0"],
    new Map([[1, result({ histogram: { mean: 0, non_finite: 2 } })]]),
  )[0];
  expect(faulty.message).toBe(
    "probs holds 2 NaN or infinite values. probs holds 2 NaN or infinite values, so its mean is not a number near 0.",
  );
});

test("stale, predicted, and snapshot-only answers are not judged", () => {
  const lines = ["probs = s.softmax(-1)  # sums(-1): 1"];
  for (const stale of [
    result({}, { fresh: false }),
    result({}, { predicted: true }),
    result({ values: [], value_source: "paged" }),
  ])
    expect(checkValueContracts(lines, new Map([[1, stale]]))).toEqual([]);
});

test("a line with shape and value contracts gets one mark, failing if either fails", () => {
  const lines = ["d = p * 2  # shape: B, T; range: 0..1"];
  const results = new Map([[1, result({ name: "d", maximum: 1.6 })]]);
  const marks = mergeChecks(
    checkContracts(lines, results),
    checkValueContracts(lines, results),
  );
  expect(marks.get(1)).toMatchObject({
    ok: false,
    text: "[B, T]; range: 0..1",
  });
  // A broken line explains only what it breaks.
  expect(marks.get(1)!.message).toBe("d reaches 1.6, above 1.");
});

test("a recorded tensor becomes a contract it keeps, its range rounded outward", () => {
  const tensor = {
    shape: [1, 16, 16],
    minimum: -0.5774768,
    maximum: 0.6711398,
    histogram: { non_finite: 0 },
  } as Parameters<typeof contractFor>[0];
  const contract = contractFor(tensor);
  expect(contract).toBe("# shape: 1, 16, 16; range: -0.58..0.68; finite");
  const results = new Map([
    [
      1,
      result({ shape: [1, 16, 16], minimum: -0.5774768, maximum: 0.6711398 }),
    ],
  ]);
  expect(checkValueContracts([`y = f(x)  ${contract}`], results)[0].ok).toBe(
    true,
  );
  expect(
    contractFor({
      ...tensor,
      minimum: 0,
      maximum: 2499,
      histogram: { non_finite: 3 },
    } as typeof tensor),
  ).toBe("# shape: 1, 16, 16; range: 0..2500");
});

test("spread, zeros and dtype contracts watch an activation's health", () => {
  expect(
    parseValueContract(
      "h = relu(x)  # zeros: ..50%; std: 0.5..2; dtype: float32",
      1,
    ),
  ).toMatchObject({
    clauses: [
      { kind: "zeros", low: null, high: 0.5 },
      { kind: "std", low: 0.5, high: 2 },
      { kind: "dtype", dtype: "float32" },
    ],
  });
  // A share of zeros reads as a fraction, a percentage, or a bare number over 1.
  expect(parseValueContract("h = x  # zeros: ..0.5", 1)).toMatchObject({
    clauses: [{ high: 0.5 }],
  });
  expect(parseValueContract("h = x  # zeros: ..50", 1)).toMatchObject({
    clauses: [{ high: 0.5 }],
  });
  const dead = result({
    name: "h",
    dtype: "float32",
    histogram: { counts: [6, 2], zeros: 6, non_finite: 0, std: 0.1, mean: 0.1 },
  });
  const check = checkValueContracts(
    ["h = relu(x)  # zeros: ..50%; std: 0.5..2; dtype: float16"],
    new Map([[1, dead]]),
  )[0];
  expect(check.ok).toBe(false);
  expect(check.message).toBe(
    "h is 75% zeros, above 50%. h's σ is 0.1, below 0.5. h is float32, not float16.",
  );
});

test("accepting the recording redraws only the broken clauses", () => {
  const relu = {
    name: "h",
    dtype: "float32",
    minimum: 0,
    maximum: 5.56,
    histogram: {
      counts: [10, 6],
      zeros: 5,
      non_finite: 0,
      std: 1.4,
      mean: 1.2,
    },
  } as unknown as Parameters<typeof acceptRecorded>[1];
  expect(
    acceptRecorded(
      "h = relu(s)  # shape: B, T; zeros: ..30%; range: 0..10",
      relu,
    ),
  ).toBe("h = relu(s)  # shape: B, T; zeros: ..35%; range: 0..10");
  // The mean allows 10% of the larger of the mean and σ (1.4 here).
  expect(
    acceptRecorded("h = relu(s)  # std: 0.1..0.2; mean: 0 ± 0.1", relu),
  ).toBe("h = relu(s)  # std: 0.7..2.8; mean: 1.2 ± 0.14");
  // A broken `finite` or `sums` cannot be reworded true, so it is dropped.
  const faulty = { ...relu, histogram: { ...relu.histogram!, non_finite: 2 } };
  expect(acceptRecorded("h = log(s)  # finite", faulty)).toBe("h = log(s)");
  expect(acceptRecorded("h = relu(s)  # range: 0..10", relu)).toBeNull();
});

test("sums on snapshot tensors are asked of the backend and answered", () => {
  const paged = result({ id: "t9", value_source: "paged", values: [] });
  const lines = ["p = s.softmax(-1)  # sums(-1): 1"];
  const results = new Map([[1, paged]]);
  expect(sumsQuestions(lines, results)).toEqual([
    { key: sumsKey("t9", -1, 1), tensorId: "t9", axis: -1, value: 1 },
  ]);
  expect(checkValueContracts(lines, results)).toEqual([]);
  const answers = new Map([
    [sumsKey("t9", -1, 1), { count: 100, off: 100, worst: 0.5 }],
  ]);
  expect(checkValueContracts(lines, results, answers)[0].message).toBe(
    "100 of 100 sums over axis -1 are not 1; one is 0.5.",
  );
});

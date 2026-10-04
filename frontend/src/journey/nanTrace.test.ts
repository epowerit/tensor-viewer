import { expect, test } from "vitest";
import type { Run } from "../api/client";
import fixture from "./fixtures/nan-trace.json";
import softmaxFixture from "./fixtures/nan-softmax.json";
import { traceNonFinite } from "./nanTrace";

// The NaN audit model on x = 0…23:
//   scaled = x - 5; logged = log(scaled); ratio = scaled / (x - 4) * 1.1
//   return logged + ratio
const trace = fixture as unknown as Run["trace"];
const read = async (tensorId: string, indices: number[]) =>
  indices.map((index) => trace.tensors[tensorId].values[index]);
const output = trace.output_ids[0];

test("a NaN from the log of a negative number is traced to the log", async () => {
  const found = await traceNonFinite(trace, output, 0, read);
  expect(found.cells.map((cell) => cell.opId)).toEqual(["op5", "op1"]);
  expect(found.cause).toBe("the logarithm of a negative number is NaN");
  expect(found.from).toEqual([
    expect.objectContaining({ tensorId: "t1", index: 0, value: -5 }),
  ]);
  expect(found.approximate).toBe(false);
});

test("−∞ from log(0), and a NaN at x = 4 is still the log's NaN", async () => {
  const minus = await traceNonFinite(trace, output, 5, read);
  expect(minus.cells.at(-1)?.opId).toBe("op1");
  expect(minus.cause).toBe("log(0) is −∞");
  // log(−1) + (−1 / 0): the NaN came from the log, not the division.
  const four = await traceNonFinite(trace, output, 4, read);
  expect(four.cells.at(-1)?.opId).toBe("op1");
});

test("an infinity from dividing by zero", async () => {
  // ratio at x = 4: −1 / 0 · 1.1 = −∞, traced from the product to the division.
  const ratio = trace.operations[4].outputs[0];
  const found = await traceNonFinite(trace, ratio, 4, read);
  expect(found.cells.map((cell) => cell.opId)).toEqual(["op4", "op3"]);
  expect(found.cause).toBe("dividing by 0 gives −∞");
});

test("a finite value has nothing to trace", async () => {
  const found = await traceNonFinite(trace, output, 6, read);
  expect(found.cells).toEqual([]);
  expect(found.stopped).toBe("This value is finite.");
});

test("a softmax over a row of −∞ makes NaN; the −∞ was written on purpose", async () => {
  // mask = x.sum(-1, keepdim=True) > 30; scores = x.masked_fill(mask, -inf)
  // weights = scores.softmax(-1); return weights * 2
  const masked = softmaxFixture as unknown as Run["trace"];
  const readMasked = async (tensorId: string, indices: number[]) =>
    indices.map((index) => masked.tensors[tensorId].values[index]);
  const found = await traceNonFinite(
    masked,
    masked.output_ids[0],
    8,
    readMasked,
  );
  expect(found.cells.map((cell) => cell.opId)).toEqual(["op4", "op3"]);
  expect(found.cause).toBe(
    "every score in this row is −∞, so softmax divides 0 by 0",
  );
  expect(found.infinity?.opId).toBe("op2");
  const infinity = await traceNonFinite(
    masked,
    found.infinity!.tensorId,
    found.infinity!.index,
    readMasked,
  );
  expect(infinity.cause).toBe(
    "masked_fill writes −∞ here on purpose, where the mask is set",
  );
});

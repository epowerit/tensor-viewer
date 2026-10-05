import { expect, test } from "vitest";
import type { Operation } from "../api/client";
import { warmTimes } from "./timing";

const op = (index: number, kind: string, duration_us: number | null) =>
  ({ id: `op${index}`, index, kind, duration_us }) as unknown as Operation;

test("a kind's first call counts at the median of its later calls", () => {
  const found = warmTimes([
    op(0, "linear", 1200),
    op(1, "softmax", 500),
    op(2, "linear", 10),
    op(3, "linear", 14),
    op(4, "linear", 30),
    op(5, "softmax", 7),
  ]);
  expect(found.times.get("op0")).toBe(14);
  expect(found.times.get("op1")).toBe(7);
  expect(found.times.get("op4")).toBe(30);
  expect(found.setup).toBe(1200 - 14 + 500 - 7);
  expect([...found.estimated]).toEqual(["op0", "op1"]);
});

test("a kind that runs once, or a first call no slower, keeps its time", () => {
  const found = warmTimes([
    op(0, "embedding", 600),
    op(1, "add", 3),
    op(2, "add", 9),
    op(3, "reshape", null),
  ]);
  expect(found.times.get("op0")).toBe(600);
  expect(found.times.get("op1")).toBe(3);
  expect(found.times.has("op3")).toBe(false);
  expect(found.setup).toBe(0);
  expect(found.estimated.size).toBe(0);
});

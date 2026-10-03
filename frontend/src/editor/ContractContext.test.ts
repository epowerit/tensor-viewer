import { expect, test } from "vitest";
import type { LineResult } from "../console/script";
import { contractIndex } from "./ContractContext";

test("checked contracts land on the tensor and step their line produced", () => {
  const line = (
    output: string,
    step: string,
    extra: Partial<LineResult> = {},
  ) =>
    ({
      operations: [{ id: step }],
      output: { id: output },
      error: null,
      fresh: true,
      ...extra,
    }) as unknown as LineResult;
  const kept = { line: 1, text: "range: 0..1", ok: true, message: "kept" };
  const unreadable = { line: 2, text: "", ok: false, message: "bad" };
  const stale = { line: 3, text: "finite", ok: false, message: "old" };
  const index = contractIndex(
    new Map([
      [1, kept],
      [2, unreadable],
      [3, stale],
    ]),
    new Map([
      [1, line("t1", "op1")],
      [2, line("t2", "op2")],
      [3, line("t3", "op3", { fresh: false })],
    ]),
  );
  expect([...index.byTensor.keys()]).toEqual(["t1"]);
  expect(index.byOperation.get("op1")).toBe(kept);
});

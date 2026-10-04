import { expect, test } from "vitest";
import type { Run } from "../api/client";
import { applyFix, fixCandidates } from "./codeFixes";

/** A run whose step `kind` on `line` failed with `message`. */
function failedRun(
  code: string,
  line: number,
  kind: string,
  message: string,
  inputs: { name: string; shape: number[]; dtype?: string }[],
): Run {
  const tensors = Object.fromEntries(
    inputs.map((t, i) => [`t${i}`, { id: `t${i}`, dtype: "float32", ...t }]),
  );
  return {
    project: { code, files: {}, entry_path: "model.py" },
    trace: {
      tensors,
      operations: [
        {
          id: "op0",
          kind,
          status: "error",
          inputs: Object.keys(tensors),
          outputs: [],
          source: { line, text: "", file: "model.py" },
        },
      ],
      error: { type: "RuntimeError", message, line, file: "model.py" },
    },
  } as unknown as Run;
}

test("a matrix product that does not line up tries each operand transposed", () => {
  const code =
    "def forward(self, view, keys):\n    scores = view @ keys\n    return scores";
  const run = failedRun(
    code,
    2,
    "__matmul__",
    "mat1 and mat2 shapes cannot be multiplied",
    [
      { name: "view", shape: [6, 4] },
      { name: "keys", shape: [6, 4] },
    ],
  );
  const fixes = fixCandidates(run, null);
  expect(fixes.map((fix) => fix.after.trim())).toEqual([
    "scores = view.transpose(-2, -1) @ keys",
    "scores = view @ keys.transpose(-2, -1)",
  ]);
  expect(applyFix(code, fixes[1])).toBe(
    "def forward(self, view, keys):\n    scores = view @ keys.transpose(-2, -1)\n    return scores",
  );
  // Code edited since: the fix no longer applies.
  expect(applyFix(code.replace("view @", "view  @"), fixes[1])).toBeNull();
});

test("view after a permute becomes reshape; a linear layer is built for its input", () => {
  const view = fixCandidates(
    failedRun(
      "y = x.permute(1, 0).view(-1)",
      1,
      "view",
      "view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.",
      [{ name: "x", shape: [2, 3] }],
    ),
    null,
  );
  expect(view.map((fix) => fix.after)).toEqual([
    "y = x.permute(1, 0).reshape(-1)",
  ]);
  const code = "self.proj = nn.Linear(32, 10)\n\ny = self.proj(x)";
  const linear = fixCandidates(
    failedRun(
      code,
      3,
      "linear",
      "mat1 and mat2 shapes cannot be multiplied (2x24 and 32x10)",
      [
        { name: "x", shape: [2, 24] },
        { name: "proj.weight", shape: [10, 32] },
      ],
    ),
    null,
  );
  expect(linear.map((fix) => [fix.line, fix.after])).toContainEqual([
    1,
    "self.proj = nn.Linear(24, 10)",
  ]);
});

test("a misspelled name takes the suggested spelling", () => {
  const run = failedRun(
    "y = frist + 1",
    1,
    "",
    "name 'frist' is not defined",
    [],
  );
  run.trace.operations = [];
  run.trace.error!.type = "NameError";
  const fixes = fixCandidates(run, {
    title: "",
    explanation: "",
    operands: [],
    suggestion: "Did you mean first?",
  });
  expect(fixes.map((fix) => fix.after)).toEqual(["y = first + 1"]);
});

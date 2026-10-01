import { expect, test } from "vitest";
import type { Run } from "../api/client";
import { collectProblems, problemCounts } from "./problems";

const run = (trace: object) =>
  ({
    project: { entry_path: "model.py" },
    trace: { operations: [], tensors: {}, error: null, ...trace },
  }) as unknown as Run;

test("a failed operation becomes a located problem with its diagnosis", () => {
  const problems = collectProblems(
    run({
      error: { type: "RuntimeError", message: "shapes", line: 4 },
      tensors: {
        a: { name: "a", shape: [6, 4], numel: 24 },
        b: { name: "b", shape: [6, 4], numel: 24 },
      },
      operations: [
        { id: "op0", status: "ok", inputs: [], kind: "view" },
        {
          id: "op1",
          status: "error",
          kind: "matmul",
          inputs: ["a", "b"],
          arguments: {},
          error: "shapes",
        },
      ],
      warnings: ["A tensor changed outside a recorded PyTorch operation."],
    }),
    { stale: true },
  );
  expect(problems.map((p) => p.severity)).toEqual(["error", "warning", "info"]);
  expect(problems[0]).toMatchObject({
    title: "The inner dimensions do not match",
    detail: "RuntimeError: shapes",
    file: "model.py",
    line: 4,
    node: "op1",
  });
  expect(problems[0].diagnosis?.suggestion).toContain("transpose");
  expect(problemCounts(problems)).toEqual({ errors: 1, warnings: 1 });
});

test("errors outside an operation, input issues, and clean runs", () => {
  const syntax = collectProblems(
    run({
      error: { type: "SyntaxError", message: "bad", line: 2, file: "a.py" },
    }),
  );
  expect(syntax[0]).toMatchObject({
    title: "SyntaxError",
    file: "a.py",
    node: null,
    diagnosis: null,
  });
  expect(
    collectProblems(null, { inputIssue: "Use 1–6 dimensions.", stale: true }),
  ).toMatchObject([{ id: "input", severity: "error" }]);
  expect(collectProblems(run({}))).toEqual([]);
  expect(
    collectProblems(
      run({ weight_check: { compatible: false, issues: ["missing key"] } }),
    )[0].detail,
  ).toBe("missing key");
});

test("a failing shape check is a located warning ahead of run problems", () => {
  const check = run({
    error: { type: "RuntimeError", message: "shapes", line: 2 },
    tensors: {
      a: { name: "a", shape: [2, 3], numel: 6 },
      b: { name: "b", shape: [4], numel: 4 },
    },
    operations: [
      {
        id: "op0",
        status: "error",
        kind: "add",
        inputs: ["a", "b"],
        arguments: {},
        error: "shapes",
      },
    ],
  });
  const problems = collectProblems(run({}), { check, stale: true });
  expect(problems.map((p) => p.id)).toEqual(["shape-check", "stale"]);
  expect(problems[0]).toMatchObject({
    severity: "warning",
    title: "Shape check: These shapes cannot broadcast",
    line: 2,
    node: null,
  });
  expect(problemCounts(problems)).toEqual({ errors: 0, warnings: 1 });
  expect(collectProblems(null, { check: run({}) })).toEqual([]);
});

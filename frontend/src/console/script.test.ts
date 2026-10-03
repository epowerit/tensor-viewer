import { expect, test } from "vitest";
import { scriptRun, type Run } from "../api/client";
import { inputIssue } from "../inputs/fixtures";
import {
  appendSnippet,
  consoleProject,
  EXAMPLES,
  knownShapes,
  lastVariable,
  lineResults,
  withoutComment,
  needsValues,
  withShapeCheck,
  parseShape,
  SNIPPETS,
} from "./script";

const header =
  "import math\n\nimport torch\nimport torch.nn.functional as F\nfrom torch import nn\n\n\nclass Console(nn.Module):\n    def forward(self, x):\n";
function recorded_run(script: string, error?: { line: number }): Run {
  const tensor = (id: string, name: string, shape: number[]) => ({
    id,
    name,
    shape,
  });
  return scriptRun({
    id: "run",
    project: {
      ...consoleProject("test"),
      script,
      code: header + script.replace(/^/gm, "        ") + "\n",
    },
    trace: {
      tensors: {
        t0: tensor("t0", "x", [2, 3, 4]),
        t1: tensor("t1", "y", [2, 12]),
        t2: tensor("t2", "z", [12, 2]),
      },
      error: error
        ? { type: "RuntimeError", message: "shape mismatch", ...error }
        : null,
      operations: [
        { id: "op0", kind: "reshape", outputs: ["t1"], source: { line: 10 } },
        { id: "op1", kind: "permute", outputs: ["t2"], source: { line: 11 } },
      ],
    },
  } as unknown as Run);
}

test("console runs are presented in script coordinates", () => {
  const run = recorded_run("y = x.reshape(2, 12)\nz = y.permute(1, 0)", {
    line: 11,
  });
  expect(run.project.code).toBe("y = x.reshape(2, 12)\nz = y.permute(1, 0)");
  expect(run.trace.operations.map((op) => op.source!.line)).toEqual([1, 2]);
  expect(run.trace.error!.line).toBe(2);
  const syntax = recorded_run("y = x.reshape(2, 12)\nz = (", {
    line: 11,
    message: "'(' was never closed (<tensorviewer-project>, line 11)",
  } as { line: number });
  expect(syntax.trace.error!.message).toBe("'(' was never closed (line 2)");
});

test("line results go stale from the first edited line onward", () => {
  const script = "y = x.reshape(2, 12)\nz = y.permute(1, 0)";
  const run = recorded_run(script, { line: 11 });
  const same = lineResults(run, script);
  expect(same.get(1)!.output!.name).toBe("y");
  expect(same.get(2)!.error).toBe("RuntimeError: shape mismatch");
  expect([...same.values()].every((r) => r.fresh)).toBe(true);
  const edited = lineResults(run, "y = x.reshape(2, 12)\nz = y.permute(0, 1)");
  expect(edited.get(1)!.fresh).toBe(true);
  expect(edited.get(2)!.fresh).toBe(false);
  expect(knownShapes(edited)).toEqual({ y: [2, 12] });
  // A comment, such as a contract, changes nothing that runs.
  const commented = lineResults(
    run,
    "y = x.reshape(2, 12)  # shape: 2, 12\nz = y.permute(1, 0)",
  );
  expect([...commented.values()].every((r) => r.fresh)).toBe(true);
  expect(withoutComment('s = "a # b"  # note')).toBe('s = "a # b"');
  expect(withoutComment("t = x  ")).toBe("t = x");
  // Inlays stay on their code as lines are added above; values below the
  // edit are stale.
  const shifted = lineResults(
    run,
    "w = x\ny = x.reshape(2, 12)\nz = y.permute(1, 0)",
  );
  // A comment line added above changes nothing that runs.
  const noted = lineResults(run, "# note\n" + script);
  expect(noted.get(2)!.output!.name).toBe("y");
  expect([...noted.values()].every((r) => r.fresh)).toBe(true);
  expect(shifted.get(2)!.output!.name).toBe("y");
  expect(shifted.get(3)!.error).toBe("RuntimeError: shape mismatch");
  expect(shifted.get(1)).toBeUndefined();
  expect(shifted.get(2)!.fresh).toBe(false);
  // A deleted line takes its result with it.
  expect(lineResults(run, "z = y.permute(1, 0)").get(1)!.error).toBe(
    "RuntimeError: shape mismatch",
  );
  expect(lineResults(run, script, true).get(1)!.fresh).toBe(false);
  expect(lineResults(null, script).size).toBe(0);
  // Operations recorded in another file never annotate this one.
  expect(lineResults(run, script, false, "helper.py").size).toBe(0);
});

test("snippets extend the latest variable with a unique name", () => {
  expect(lastVariable("a = x\n  b = 2\nc == 3\n# d = 1")).toBe("a");
  expect(lastVariable("x.permute(1, 0)", "q")).toBe("q");
  const permute = SNIPPETS.find((s) => s.id === "permute")!;
  const once = appendSnippet("y = x + 1\n\n", permute, "x", { y: [2, 3, 4] });
  expect(once).toBe("y = x + 1\npermuted = y.permute(2, 1, 0)");
  expect(appendSnippet(once, permute, "x").split("\n")[2]).toBe(
    "permuted2 = permuted.permute(1, 0)",
  );
  expect(appendSnippet("", permute, "x")).toBe("permuted = x.permute(1, 0)");
});

test("snippets stay reachable before the built-in views example's explicit return", () => {
  const views = EXAMPLES.find((item) => item.id === "views")!;
  const clone = SNIPPETS.find((item) => item.id === "clone")!;
  expect(appendSnippet(views.script, clone, "x")).toBe(
    views.script.replace(
      "return view, frozen, x",
      "copy = frozen.clone()\nreturn view, frozen, x",
    ),
  );
});

test("snippet insertion preserves multiline returns and trailing comments", () => {
  const clone = SNIPPETS.find((item) => item.id === "clone")!;
  const script =
    "y = x + 1\nreturn (\n    y,\n    x,\n)\n# These outputs are intentional.";
  expect(appendSnippet(script, clone, "x")).toBe(
    "y = x + 1\ncopy = y.clone()\nreturn (\n    y,\n    x,\n)\n# These outputs are intentional.",
  );
  expect(appendSnippet("return x", clone, "x")).toBe(
    "copy = x.clone()\nreturn x",
  );
});

test("quoted return and assignment text does not steer snippet insertion", () => {
  const clone = SNIPPETS.find((item) => item.id === "clone")!;
  const script = 'note = """\nreturn x\nphantom = x\n"""\ny = x + 1\nreturn y';
  expect(appendSnippet(script, clone, "x")).toBe(
    'note = """\nreturn x\nphantom = x\n"""\ny = x + 1\ncopy = y.clone()\nreturn y',
  );
  const literal = 'y = """\nphantom = x\n"""';
  expect(lastVariable(literal)).toBe("y");
  const nested = "def helper():\n    return x\ny = x + 1\n# return y";
  expect(appendSnippet(nested, clone, "x")).toBe(`${nested}\ncopy = y.clone()`);
});

test("unreachable assignments after a return are not used as snippet operands", () => {
  const clone = SNIPPETS.find((item) => item.id === "clone")!;
  expect(appendSnippet("y = x + 1\nreturn y\nnever = x + 2", clone, "x")).toBe(
    "y = x + 1\ncopy = y.clone()\nreturn y\nnever = x + 2",
  );
});

test("shape entry accepts common spellings and explains mistakes", () => {
  expect(parseShape("2, 3, 4")).toEqual([2, 3, 4]);
  expect(parseShape("[2 x 3]")).toEqual([2, 3]);
  expect(typeof parseShape("")).toBe("string");
  expect(typeof parseShape("2, 0")).toBe("string");
  expect(typeof parseShape("1,1,1,1,1,1,1")).toBe("string");
  expect(typeof parseShape("2.5")).toBe("string");
});

test("every example is a valid starting project", () => {
  for (const example of EXAMPLES) {
    const draft = consoleProject("x", example);
    expect(draft.script).toBe(example.script);
    expect(inputIssue(draft.input)).toBe("");
    expect(
      !draft.input.axis_names.length ||
        draft.input.axis_names.length === draft.input.shape.length,
    ).toBe(true);
  }
});

test("a shape check fills only the lines the run no longer covers", () => {
  const first = "y = x.reshape(2, 12)\nz = y.permute(1, 0)";
  const edited = "y = x.reshape(2, 12)\nz = y.permute(0, 1)";
  const recorded = lineResults(recorded_run(first), edited);
  const checked = lineResults(recorded_run(edited), edited);
  const merged = withShapeCheck(recorded, checked);
  expect(merged.get(1)).toMatchObject({ fresh: true });
  expect(merged.get(1)!.predicted).toBeUndefined();
  expect(merged.get(1)!.operations).toHaveLength(1);
  expect(merged.get(2)).toMatchObject({
    fresh: true,
    predicted: true,
    operations: [],
  });
  expect(merged.get(2)!.output!.name).toBe("z");
  // A check of older text never overrides anything.
  const outdated = lineResults(recorded_run(first), edited);
  expect(withShapeCheck(recorded, outdated).get(2)!.fresh).toBe(false);
  expect(withShapeCheck(new Map(), checked).get(1)!.predicted).toBe(true);
});

test("a step whose shape depends on values limits the check without an error", () => {
  expect(
    needsValues({ type: "NotImplementedError", message: "Could not run" }),
  ).toBe(true);
  expect(
    needsValues({
      type: "RuntimeError",
      message: "Cannot copy out of meta tensor; no data!",
    }),
  ).toBe(true);
  expect(
    needsValues({
      type: "RuntimeError",
      message: "shapes cannot be multiplied",
    }),
  ).toBe(false);
  expect(needsValues(null)).toBe(false);
  const script = "m = x > 1\npicked = x[m]";
  const checked = lineResults(
    {
      ...recorded_run(script),
      trace: {
        ...recorded_run(script).trace,
        operations: [],
        error: {
          type: "NotImplementedError",
          message: "Could not run aten::nonzero with meta tensors",
          line: 2,
        },
      },
    } as never,
    script,
  );
  expect(withShapeCheck(new Map(), checked).get(2)).toMatchObject({
    error: null,
    output: null,
    predicted: true,
    needsValues: true,
  });
});

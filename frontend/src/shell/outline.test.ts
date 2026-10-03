import { describe, expect, it, test } from "vitest";
import type { Operation } from "../api/client";
import type { PlaybackStep } from "../journey/loops";
import { OTHER_CODE, expression, foldedOutline, stepOutline } from "./outline";

const op = (id: string, line: number | null, module = "", file?: string) =>
  ({
    id,
    index: Number(id.slice(2)),
    kind: "linear",
    outputs: [],
    module,
    source: line === null ? null : { line, file: file ?? null },
  }) as unknown as Operation;

const steps = (ops: Operation[]): PlaybackStep[] =>
  ops.map((operation) => ({ id: operation.id, operation }));

test("steps sit under the file whose lines recorded them, in run order", () => {
  const ops = [
    op("op0", 3),
    op("op1", 12, "", "layers.py"),
    op("op2", 4),
    op("op3", 9, "", "site-packages/torch/nn.py"),
    op("op4", null),
  ];
  const outline = stepOutline(
    steps(ops),
    ops,
    ["model.py", "layers.py"],
    "model.py",
  );
  const ids = (key: string) =>
    outline
      .get(key)!
      .flatMap((row) => (row.kind === "step" ? [row.step.id] : []));
  expect(ids("model.py")).toEqual(["op0", "op2"]);
  expect(ids("layers.py")).toEqual(["op1"]);
  // Code outside the project, or with no source, gathers apart.
  expect(ids(OTHER_CODE)).toEqual(["op3", "op4"]);
  expect(outline.get("model.py")![0]).toMatchObject({ kind: "step", line: 3 });
});

test("a block's own step between its children keeps its block", () => {
  const ops = [
    op("op0", 1, "GPT / blocks.0 / blocks.0.attention"),
    op("op1", 2, "GPT / blocks.0 / blocks.0.attention"),
    op("op2", 3, "GPT / blocks.0"),
    op("op3", 4, "GPT / blocks.0 / blocks.0.norm2"),
    op("op4", 5, "GPT / blocks.0 / blocks.0.mlp"),
    op("op5", 6, "GPT / blocks.0 / blocks.0.mlp"),
  ];
  const rows = stepOutline(steps(ops), ops, ["model.py"], "model.py").get(
    "model.py",
  )!;
  expect(
    rows.map((row) =>
      row.kind === "module" ? `# ${row.name}` : `${row.step.id}@${row.depth}`,
    ),
  ).toEqual([
    "# attention",
    "op0@3",
    "op1@3",
    // The residual add is the block's own, not the model's.
    "op2@2",
    // The one-step norm reads as part of the block, with no header.
    "op3@2",
    "# mlp",
    "op4@3",
    "op5@3",
  ]);
});

test("a folded loop sits in the call that loops", () => {
  const ops = [
    op("op0", 10, "GPT / blocks.0 / blocks.0.norm"),
    op("op1", 10, "GPT / blocks.1 / blocks.1.norm"),
  ];
  const loop = {
    id: "loop:x",
    fold: {
      id: "loop:x",
      line: 64,
      file: null,
      text: "for block in self.blocks",
      iterations: [["op0"], ["op1"]],
    },
  } as unknown as PlaybackStep;
  const rows = stepOutline([loop], ops, ["model.py"], "model.py").get(
    "model.py",
  )!;
  expect(rows[0]).toMatchObject({ kind: "module", path: "GPT" });
  expect(rows[1]).toMatchObject({ kind: "step", depth: 1, line: 64 });
});

test("module headers mark blocks; one-step layer calls read as their parent", () => {
  const ops = [
    op("op0", 1, "GPT"),
    op("op1", 2, "GPT / blocks.0 / blocks.0.norm1"),
    op("op2", 3, "GPT / blocks.0 / blocks.0.attention"),
    op("op3", 4, "GPT / blocks.0 / blocks.0.attention"),
    op("op4", 5, "GPT"),
  ];
  const rows = stepOutline(steps(ops), ops, ["model.py"], "model.py").get(
    "model.py",
  )!;
  expect(
    rows.map((row) => (row.kind === "module" ? `# ${row.name}` : row.step.id)),
  ).toEqual([
    "# GPT",
    "op0",
    "# blocks.0",
    "op1",
    "# attention",
    "op2",
    "op3",
    // Back in GPT: no new header, the step steps back out.
    "op4",
  ]);
  // Steps indent under their module.
  expect(
    rows.find((row) => row.kind === "step" && row.step.id === "op2"),
  ).toMatchObject({
    depth: 3,
  });
});

describe("expression", () => {
  it("drops an assignment's target", () => {
    expect(expression("q = self.split(self.query(x))")).toBe(
      "self.split(self.query(x))",
    );
    expect(expression("q, k = x.chunk(2, dim=-1)")).toBe("x.chunk(2, dim=-1)");
    expect(expression("x += self.mlp(x)")).toBe("self.mlp(x)");
    expect(
      expression("q = x.transpose(1, 2)  # axes: batch, heads, tokens"),
    ).toBe("x.transpose(1, 2)");
  });
  it("keeps statements that assign nothing", () => {
    expect(expression("return self.head(x)")).toBe("return self.head(x)");
    expect(expression("mask == 0")).toBe("mask == 0");
    expect(expression("f(a=1)")).toBe("f(a=1)");
    expect(expression("out[0] = x")).toBe("out[0] = x");
    expect(expression('print("#1")')).toBe('print("#1")');
  });
});

describe("foldedOutline", () => {
  const step = (id: string, depth: number) =>
    ({
      kind: "step",
      step: { id } as PlaybackStep,
      line: 1,
      depth,
    }) as const;
  const module = (path: string) =>
    ({
      kind: "module",
      name: path.split(" / ").at(-1)!,
      path,
      depth: path.split(" / ").length - 1,
    }) as const;
  const rows = [
    module("GPT"),
    step("embed", 1),
    module("GPT / blocks.0"),
    step("norm", 2),
    module("GPT / blocks.0 / attention"),
    step("q", 3),
    step("k", 3),
    module("GPT / blocks.0"),
    step("norm again", 2),
    step("head", 1),
  ];
  const ids = (shown: ReturnType<typeof foldedOutline>) =>
    shown.map(({ row, hidden }) =>
      row.kind === "module"
        ? `[${row.path}${hidden !== undefined ? ` −${hidden}` : ""}]`
        : row.step.id,
    );
  it("hides what a folded header holds, nested headers too", () => {
    expect(ids(foldedOutline(rows, new Set(["GPT / blocks.0"])))).toEqual([
      "[GPT]",
      "embed",
      "[GPT / blocks.0 −3]",
      "[GPT / blocks.0]",
      "norm again",
      "head",
    ]);
  });
  it("keys a repeated call by its count", () => {
    const shown = foldedOutline(rows, new Set(["GPT / blocks.0#2"]));
    // The second call folds; the first, and its attention, stay open.
    expect(ids(shown).slice(2)).toEqual([
      "[GPT / blocks.0]",
      "norm",
      "[GPT / blocks.0 / attention]",
      "q",
      "k",
      "[GPT / blocks.0 −1]",
      "head",
    ]);
    expect(shown.filter((item) => item.key).map((item) => item.key)).toEqual([
      "GPT",
      "GPT / blocks.0",
      "GPT / blocks.0 / attention",
      "GPT / blocks.0#2",
    ]);
  });
  it("keeps a folded block open while it holds the current step", () => {
    const shown = foldedOutline(
      rows,
      new Set(["GPT / blocks.0"]),
      (row) => row.kind === "step" && row.step.id === "q",
    );
    expect(ids(shown)).toContain("q");
  });
});

import { describe, expect, it } from "vitest";
import type { LoopStep, Operation, Run, Tensor } from "../api/client";
import { buildJourney } from "./graph";
import {
  displayedOperation,
  foldJourney,
  locateOperation,
  loopFolds,
  loopLines,
  passLoops,
  passRanges,
  playbackSteps,
} from "./loops";

const tensor = (id: string, role: Tensor["role"] = "intermediate"): Tensor => ({
  id,
  name: id,
  role,
  shape: [2, 4],
  axes: ["batch", "features"],
  dtype: "float32",
  strides: [4, 1],
  storage_id: id,
  storage_offset: 0,
  contiguous: true,
  numel: 8,
  values: [],
  minimum: 0,
  maximum: 0,
});
const loop = (iteration: number, id = "/0:12"): LoopStep => ({
  id,
  line: 12,
  file: null,
  text: "for block in self.blocks",
  iteration,
});
function op(
  index: number,
  kind: string,
  line: number,
  inputs: string[],
  output: string,
  loops: LoopStep[] = [],
): Operation {
  return {
    id: `op${index}`,
    index,
    kind,
    function: `torch.${kind}`,
    inputs,
    outputs: [output],
    arguments: {},
    source: { line, text: kind },
    module: "Model",
    status: "ok",
    error: null,
    loops,
    lesson: {
      title: kind,
      summary: kind,
      detail: kind,
      category: "compute",
      interaction: "inspect",
      mapping: null,
      axis_order: null,
    },
  };
}
// x → scale → [linear → relu] × 3 → sum, with a mask made inside every pass.
function looped(): Run["trace"] {
  const operations = [op(0, "mul", 10, ["x"], "s")];
  let carried = "s";
  for (let pass = 1; pass <= 3; pass++) {
    const base = operations.length;
    operations.push(
      op(base, "ones", 14, [], `m${pass}`, [loop(pass)]),
      op(base + 1, "linear", 15, [carried, `m${pass}`], `l${pass}`, [
        loop(pass),
      ]),
      op(base + 2, "relu", 16, [`l${pass}`], `r${pass}`, [loop(pass)]),
    );
    carried = `r${pass}`;
  }
  operations.push(op(operations.length, "sum", 20, [carried], "y"));
  const ids = [
    "s",
    "y",
    ...[1, 2, 3].flatMap((p) => [`m${p}`, `l${p}`, `r${p}`]),
  ];
  return {
    schema_version: "1",
    operations,
    tensors: Object.fromEntries(
      [tensor("x", "input"), ...ids.map((id) => tensor(id))].map((t) => [
        t.id,
        t,
      ]),
    ),
    input_ids: ["x"],
    output_ids: ["y"],
    error: null,
    stdout: "",
    duration_ms: 0,
  };
}

describe("folded loops", () => {
  it("folds a loop whose iterations repeat the same work", () => {
    const folds = loopFolds(looped());
    expect(folds).toHaveLength(1);
    expect(folds[0].iterations).toEqual([
      ["op1", "op2", "op3"],
      ["op4", "op5", "op6"],
      ["op7", "op8", "op9"],
    ]);
  });

  it("keeps a loop whose iterations differ unfolded", () => {
    const trace = looped();
    trace.operations[5] = { ...trace.operations[5], kind: "gelu" };
    expect(loopFolds(trace)).toEqual([]);
  });

  it("plays the first iteration, then one step for the repeats", () => {
    const trace = looped();
    const steps = playbackSteps(trace.operations, loopFolds(trace));
    expect(steps.map((step) => step.id)).toEqual([
      "op0",
      "op1",
      "op2",
      "op3",
      "loop:/0:12",
      "op10",
    ]);
  });

  it("draws the body once, without the edges that run backwards", () => {
    const trace = looped();
    const folds = loopFolds(trace);
    const graph = foldJourney(buildJourney(trace), folds, {}, trace);
    const ids = graph.nodes.map((node) => node.id);
    expect(ids).toEqual(["input-x", "op0", "op1", "op2", "op3", "op10"]);
    const edges = graph.edges.map((edge) => `${edge.source}>${edge.target}`);
    // The value leaving the last pass reaches the sum from the body's end.
    expect(edges).toContain("op3>op10");
    expect(edges).not.toContain("op3>op2");
    expect(new Set(edges).size).toBe(edges.length);
    // The mask sits beside its consumer, inside the body, not at the left.
    const depth = (id: string) => graph.nodes.find((n) => n.id === id)!.depth;
    expect(depth("op1")).toBe(depth("op2") - 1);
    expect(graph.loops?.[0]).toMatchObject({ shown: 1 });
  });

  it("shows the chosen iteration's operations and tensors in the body", () => {
    const trace = looped();
    const folds = loopFolds(trace);
    const graph = foldJourney(
      buildJourney(trace),
      folds,
      { "/0:12": 3 },
      trace,
    );
    const body = graph.nodes.find((node) => node.id === "op3")!;
    expect(body.operation?.id).toBe("op9");
    expect(body.tensors.map((t) => t.id)).toEqual(["r3"]);
    const out = graph.edges.find((edge) => edge.target === "op10")!;
    expect(out.tensorId).toBe("r3");
    expect(displayedOperation(folds, { "/0:12": 2 }, "op2")).toBe("op5");
  });

  it("finds a later iteration's operation on screen", () => {
    const folds = loopFolds(looped());
    expect(locateOperation(folds, "op8")).toEqual({
      id: "op2",
      shown: { "/0:12": 3 },
    });
    expect(locateOperation(folds, "op0")).toEqual({ id: "op0", shown: {} });
  });

  it("reports what each pass leaves behind", () => {
    const trace = looped();
    trace.tensors.r2 = { ...trace.tensors.r2, minimum: -2, maximum: 5 };
    trace.tensors.r3 = { ...trace.tensors.r3, minimum: null };
    const ranges = passRanges(loopFolds(trace)[0], trace);
    expect(ranges).toEqual([
      { name: "r1", min: 0, max: 0 },
      { name: "r2", min: -2, max: 5 },
      null,
    ]);
  });

  it("marks loop headers in the editor, folded or shown in full", () => {
    const trace = looped();
    const lines = loopLines(trace, loopFolds(trace), (file) => file === null);
    expect([...lines]).toEqual([
      [
        12,
        {
          id: "/0:12",
          text: "for block in self.blocks",
          passes: 3,
          folded: true,
          firstOperation: "op1",
        },
      ],
    ]);
    trace.operations[5] = { ...trace.operations[5], kind: "gelu" };
    expect(
      loopLines(trace, loopFolds(trace), () => true).get(12),
    ).toMatchObject({ passes: 3, folded: false });
    expect(loopLines(trace, [], (file) => file === "other.py").size).toBe(0);
  });

  it("outlines the passes of a loop that is shown in full", () => {
    const trace = looped();
    expect(passLoops(trace, loopFolds(trace))).toEqual([]);
    trace.operations[5] = { ...trace.operations[5], kind: "gelu" };
    const passes = passLoops(trace, loopFolds(trace));
    expect(passes).toHaveLength(1);
    expect(passes[0].iterations).toEqual([
      ["op1", "op2", "op3"],
      ["op4", "op5", "op6"],
      ["op7", "op8", "op9"],
    ]);
  });
});

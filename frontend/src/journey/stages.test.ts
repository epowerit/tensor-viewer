import { describe, expect, it } from "vitest";
import type { ModuleCall, Operation, Run, Tensor } from "../api/client";
import { buildJourney, NODE_WIDTH } from "./graph";
import { collapseJourney, journeyStages, stageAncestors } from "./stages";

function fixture(): Run {
  const tensors = Object.fromEntries(
    ["x", "a", "b", "c", "d", "y"].map((id) => [
      id,
      {
        id,
        name: id,
        shape: [2, 3],
        axes: ["batch", "features"],
        numel: 6,
        role: id === "x" ? "input" : "intermediate",
        dtype: "float32",
        strides: [3, 1],
        storage_id: id,
        storage_offset: 0,
        contiguous: true,
        values: [0, 1, 2, 3, 4, 5],
        minimum: 0,
        maximum: 5,
      } as Tensor,
    ]),
  );
  const specs = [
    ["x", "a"],
    ["a", "b"],
    ["x", "c"],
    ["c", "d"],
    ["b", "d", "y"],
  ];
  const operations = specs.map(
    (ids, index) =>
      ({
        id: `op${index}`,
        index,
        kind: "add",
        function: "torch.add",
        module: "Model",
        inputs: ids.slice(0, -1),
        outputs: ids.slice(-1),
        arguments: {},
        source: null,
        status: "ok",
        error: null,
        lesson: {
          title: "Add",
          summary: "Add",
          detail: "Add",
          category: "compute",
          interaction: "inspect",
          mapping: null,
          axis_order: null,
        },
      }) as Operation,
  );
  const call = (
    id: string,
    parent: string | null,
    start: number,
    end: number,
    inputs: string[],
    outputs: string[],
  ): ModuleCall => ({
    id,
    parent_id: parent,
    path: parent ? "shared" : "Model",
    module_type: parent ? "Block" : "Model",
    start_index: start,
    end_index: end,
    inputs,
    outputs,
  });
  return {
    id: "run",
    project_id: "project",
    created_at: "2026-09-28",
    project: {
      name: "Stages",
      class_name: "Model",
      code: "",
      constructor: {},
      input: {
        shape: [2, 3],
        axis_names: [],
        dtype: "float32",
        seed: 7,
        generator: "arange",
      },
    },
    trace: {
      schema_version: "1",
      tensors,
      operations,
      input_ids: ["x"],
      output_ids: ["y"],
      stdout: "",
      duration_ms: 0,
      error: null,
      module_calls: [
        call("root", null, 0, 5, ["x"], ["y"]),
        call("left", "root", 0, 2, ["x"], ["b"]),
        call("right", "root", 2, 4, ["x"], ["d"]),
      ],
    },
  };
}

describe("recorded stage folding", () => {
  it("preserves fork/join tensor dependencies across repeated invocations", () => {
    const run = fixture(),
      full = buildJourney(run.trace),
      stages = journeyStages(run);
    const graph = collapseJourney(
      full,
      stages,
      new Set(["stage-left", "stage-right"]),
      run,
    );
    expect(graph.nodes.map((n) => n.id)).toEqual([
      "input-x",
      "stage-left",
      "stage-right",
      "op4",
    ]);
    expect(graph.edges.map((e) => [e.source, e.target, e.tensorId])).toEqual([
      ["input-x", "stage-left", "x"],
      ["input-x", "stage-right", "x"],
      ["stage-left", "op4", "b"],
      ["stage-right", "op4", "d"],
    ]);
    expect(
      graph.nodes.filter((n) => n.stage).map((n) => n.tensors[0].id),
    ).toEqual(["b", "d"]);
    for (const edge of graph.edges) {
      const a = graph.nodes.find((n) => n.id === edge.source)!,
        b = graph.nodes.find((n) => n.id === edge.target)!;
      expect(a.x + NODE_WIDTH).toBeLessThan(b.x);
    }
    expect(collapseJourney(full, stages, new Set(), run)).toBe(full);
    expect(full.nodes).toHaveLength(6);
  });
  it("folds only the outermost selected call and retains children for later expansion", () => {
    const run = fixture(),
      full = buildJourney(run.trace),
      stages = journeyStages(run);
    expect(
      collapseJourney(
        full,
        stages,
        new Set(stages.map((s) => s.id)),
        run,
      ).nodes.map((n) => n.id),
    ).toEqual(["input-x", "stage-root"]);
    expect(stageAncestors(stages, "op3").map((s) => s.id)).toEqual([
      "stage-root",
      "stage-right",
    ]);
    expect(stageAncestors(stages, "op4").map((s) => s.id)).toEqual([
      "stage-root",
    ]);
  });
  it("does not merge nested calls that span identical operation ranges", () => {
    const run = fixture();
    run.trace.module_calls![1].end_index = 5;
    run.trace.module_calls!.splice(2);
    const stages = journeyStages(run),
      full = buildJourney(run.trace);
    expect(stages[1].parentStageId).toBe("stage-root");
    expect(
      collapseJourney(
        full,
        stages,
        new Set(stages.map((s) => s.id)),
        run,
      ).nodes.map((n) => n.id),
    ).toEqual(["input-x", "stage-root"]);
    expect(
      collapseJourney(full, stages, new Set(["stage-left"]), run).nodes.map(
        (n) => n.id,
      ),
    ).toEqual(["input-x", "stage-left"]);
  });
  it("keeps all returned tensors and crossing intermediates, including duplicate operands", () => {
    const run = fixture();
    run.trace.module_calls![1].outputs = ["a", "b", "b"];
    run.trace.operations[4].inputs = ["a", "b", "b", "d"];
    const graph = collapseJourney(
      buildJourney(run.trace),
      journeyStages(run),
      new Set(["stage-left"]),
      run,
    );
    expect(graph.nodes.find((n) => n.stage)?.tensors.map((t) => t.id)).toEqual([
      "a",
      "b",
      "b",
    ]);
    expect(
      graph.edges
        .filter((e) => e.source === "stage-left")
        .map((e) => [e.tensorId, e.inputIndex]),
    ).toEqual([
      ["a", 0],
      ["b", 1],
      ["b", 2],
    ]);
  });
  it("keeps failed calls and does not invent a successful output", () => {
    const run = fixture();
    run.trace.operations = run.trace.operations.slice(0, 2);
    run.trace.operations[1].status = "error";
    run.trace.operations[1].outputs = [];
    run.trace.output_ids = [];
    run.trace.error = { type: "RuntimeError", message: "failed", line: null };
    run.trace.module_calls = run.trace
      .module_calls!.slice(0, 2)
      .map((c) => ({ ...c, end_index: 2, outputs: [] }));
    const stages = journeyStages(run),
      graph = collapseJourney(
        buildJourney(run.trace),
        stages,
        new Set(["stage-root"]),
        run,
      );
    expect(graph.nodes[1].stage?.failed).toBe(true);
    expect(graph.nodes[1].tensors).toEqual([]);
    expect(graph.nodes[1].terminal).toBe(false);
  });
  it("leaves older traces ungrouped and skips a generated wrapper", () => {
    const run = fixture();
    const calls = run.trace.module_calls;
    delete run.trace.module_calls;
    expect(journeyStages(run)).toEqual([]);
    run.trace.module_calls = calls;
    run.project.blueprint = { has_input: true, components: [] };
    expect(journeyStages(run).map((s) => [s.id, s.parentStageId])).toEqual([
      ["stage-left", null],
      ["stage-right", null],
    ]);
  });
});

it("labels known ViT stages without renaming arbitrary custom-code modules", () => {
  const run = fixture();
  run.project.blueprint = {
    has_input: true,
    components: [{ id: "vit", kind: "vit", parameters: {} }],
  };
  const call = run.trace.module_calls![1];
  for (const [part, title] of Object.entries({
    patches: "Patch embedding",
    tokens: "Class token + positions",
    encoder: "Transformer encoder",
    readout: "Class-token classifier",
  })) {
    call.path = `stage_0.${part}`;
    expect(
      journeyStages(run).find((stage) => stage.id === "stage-left")?.title,
    ).toBe(title);
  }
  delete run.project.blueprint;
  expect(
    journeyStages(run).find((stage) => stage.id === "stage-left")?.title,
  ).toBe("Block");
});

it("keeps external mutation effects visible when a module returns no tensor", () => {
  const run = fixture();
  run.trace.operations[1].outputs = [];
  run.trace.operations[1].mutations = [
    { before: "x", after: "b", kind: "alias" },
  ];
  run.trace.module_calls![1].outputs = [];
  run.trace.operations[4].inputs = ["b", "d"];
  const stages = journeyStages(run);
  const graph = collapseJourney(
    buildJourney(run.trace),
    stages,
    new Set(["stage-left"]),
    run,
  );
  const stage = graph.nodes.find((n) => n.id === "stage-left")!;
  expect(stage.tensors.map((t) => t.id)).toContain("b");
  expect(
    graph.edges.some(
      (e) =>
        e.source === "stage-left" && e.target === "op4" && e.tensorId === "b",
    ),
  ).toBe(true);
  expect(graph.edges.every((e) => e.source !== e.target)).toBe(true);
  expect(stage.stage?.outputs).toEqual([]);
});

it("retains an operand and a storage dependency carrying the same tensor into a stage", () => {
  const run = fixture();
  run.trace.operations[1].mutations = [
    { before: "x", after: "b", kind: "alias" },
  ];
  const graph = collapseJourney(
    buildJourney(run.trace),
    journeyStages(run),
    new Set(["stage-left"]),
    run,
  );
  expect(
    graph.edges
      .filter(
        (edge) =>
          edge.source === "input-x" &&
          edge.target === "stage-left" &&
          edge.tensorId === "x",
      )
      .map((edge) => edge.kind),
  ).toEqual(["operand", "storage"]);
});

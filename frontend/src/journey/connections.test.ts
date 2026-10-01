import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import { blankProject } from "../builder/model";
import { buildJourney } from "./graph";
import { collapseJourney, type JourneyStage } from "./stages";
import { journeyConnections, returnedTensorIds } from "./connections";

function op(index: number, inputs: string[], outputs: string[]): Operation {
  return {
    id: `op-${index}`,
    index,
    kind: "add",
    function: "torch.add",
    inputs,
    outputs,
    arguments: {},
    source: null,
    module: "Model",
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
  };
}

function trace(operations: Operation[], outputs: string[] = []): Run["trace"] {
  const ids = new Set([
    "x",
    ...operations.flatMap((operation) => [
      ...operation.inputs,
      ...operation.outputs,
      ...(operation.mutations ?? []).flatMap((mutation) => [
        mutation.before,
        mutation.after,
      ]),
    ]),
  ]);
  const tensors = Object.fromEntries(
    [...ids].map((id) => [
      id,
      {
        id,
        name: id,
        role: id === "x" ? "input" : "intermediate",
        shape: [2],
        axes: ["features"],
        strides: [1],
        dtype: "float32",
        storage_id: id,
        storage_offset: 0,
        contiguous: true,
        numel: 2,
        values: [0, 1],
        minimum: 0,
        maximum: 1,
      } satisfies Tensor,
    ]),
  );
  return {
    schema_version: "1",
    operations,
    tensors,
    input_ids: ["x"],
    output_ids: outputs,
    error: null,
    stdout: "",
    duration_ms: 0,
  };
}

describe("focused tensor connections", () => {
  it("follows forks and joins instead of the next operation in execution order", () => {
    const recorded = trace([
      op(0, ["x"], ["q"]),
      op(1, ["x"], ["k"]),
      op(2, ["q", "k"], ["scores"]),
      op(3, ["x"], ["unrelated"]),
    ]);
    const graph = buildJourney(recorded);
    graph.edges.reverse();
    expect(
      journeyConnections(graph, "input-x", recorded).outgoing.map(
        (entry) => entry.nodeId,
      ),
    ).toEqual(["op-0", "op-1", "op-3"]);
    expect(
      journeyConnections(graph, "op-0", recorded).outgoing.map(
        (entry) => entry.nodeId,
      ),
    ).toEqual(["op-2"]);
    expect(
      journeyConnections(graph, "op-2", recorded).incoming.map((entry) => [
        entry.nodeId,
        entry.tensorId,
        entry.operandRole,
      ]),
    ).toEqual([
      ["op-0", "q", "Input 1"],
      ["op-1", "k", "Input 2"],
    ]);
  });

  it("groups repeated operands but preserves their slots", () => {
    const recorded = trace([op(0, ["x", "x"], ["squared"])]);
    const graph = buildJourney(recorded);
    graph.edges.push({ ...graph.edges[0], id: "duplicate" });
    const { incoming } = journeyConnections(graph, "op-0", recorded);
    expect(incoming).toHaveLength(1);
    expect(incoming[0]).toMatchObject({
      nodeId: "input-x",
      tensorId: "x",
      inputIndices: [0, 1],
      operandRole: "Inputs 1, 2",
    });
    expect(
      journeyConnections(graph, "input-x", recorded).outgoing[0].inputIndices,
    ).toEqual([0, 1]);
  });

  it("keeps different output tensors connected to the same consumer distinct", () => {
    const recorded = trace([
      op(0, ["x"], ["a", "b"]),
      op(1, ["b", "a"], ["joined"]),
    ]);
    const connections = journeyConnections(
      buildJourney(recorded),
      "op-0",
      recorded,
    ).outgoing;
    expect(
      connections.map((entry) => [entry.tensorId, entry.inputIndices]),
    ).toEqual([
      ["b", [0]],
      ["a", [1]],
    ]);
    expect(new Set(connections.map((entry) => entry.key)).size).toBe(2);
  });

  it("keeps observed inputs for a failed operation without inventing an output", () => {
    const failed = {
      ...op(1, ["a"], []),
      status: "error" as const,
      error: "Invalid shape",
    };
    const recorded = trace([op(0, ["x"], ["a"]), failed]);
    const connections = journeyConnections(
      buildJourney(recorded),
      failed.id,
      recorded,
    );
    expect(connections.incoming.map((entry) => entry.tensorId)).toEqual(["a"]);
    expect(connections.outgoing).toEqual([]);
  });

  it("does not stop following an intermediate merely because it is also returned", () => {
    const recorded = trace(
      [op(0, ["x"], ["a"]), op(1, ["a"], ["b"])],
      ["a", "b"],
    );
    const graph = buildJourney(recorded);
    expect(graph.nodes.find((node) => node.id === "op-0")?.terminal).toBe(true);
    expect(
      journeyConnections(graph, "op-0", recorded).outgoing.map(
        (entry) => entry.nodeId,
      ),
    ).toEqual(["op-1"]);
  });

  it("keeps storage dependencies separate from tensor operands", () => {
    const recorded = trace([op(0, ["x"], ["a"]), op(1, ["a"], ["updated"])]);
    const graph = buildJourney(recorded);
    graph.edges.push({
      id: "storage-dependency",
      source: "op-0",
      target: "op-1",
      tensorId: "a",
      inputIndex: 1,
      kind: "storage",
    });
    const incoming = journeyConnections(graph, "op-1", recorded).incoming;
    expect(incoming.map((entry) => [entry.kind, entry.operandRole])).toEqual([
      ["operand", "Input 1"],
      ["storage", "Shared storage"],
    ]);
    expect(new Set(incoming.map((entry) => entry.key)).size).toBe(2);
  });

  it("uses each observed producer when an operation reuses a tensor ID", () => {
    const recorded = trace([
      op(0, ["x"], ["a"]),
      op(1, ["a"], ["a"]),
      op(2, ["a"], ["y"]),
    ]);
    const graph = buildJourney(recorded);
    expect(journeyConnections(graph, "op-1", recorded).incoming[0].nodeId).toBe(
      "op-0",
    );
    expect(journeyConnections(graph, "op-1", recorded).outgoing[0].nodeId).toBe(
      "op-2",
    );
    expect(journeyConnections(graph, "op-0", recorded).outgoing[0].nodeId).toBe(
      "op-1",
    );
  });

  it("uses visible collapsed stage neighbors without claiming module argument positions", () => {
    const recorded = trace([
      op(0, ["x"], ["a"]),
      op(1, ["a"], ["b"]),
      op(2, ["b"], ["y"]),
    ]);
    const run: Run = {
      id: "run",
      project_id: "project",
      created_at: "today",
      project: blankProject("Stages"),
      trace: recorded,
    };
    const stage: JourneyStage = {
      id: "stage-block",
      parent_id: null,
      parentStageId: null,
      path: "block",
      module_type: "Block",
      title: "Block",
      start_index: 0,
      end_index: 2,
      inputs: ["x"],
      outputs: ["b"],
      operationIds: ["op-0", "op-1"],
      failed: false,
    };
    const graph = collapseJourney(
      buildJourney(recorded),
      [stage],
      new Set([stage.id]),
      run,
    );
    expect(
      journeyConnections(graph, "input-x", recorded).outgoing[0],
    ).toMatchObject({ nodeId: stage.id, operandRole: "Used within stage" });
    expect(
      journeyConnections(graph, stage.id, recorded).incoming[0].operandRole,
    ).toBe("Used within stage");
    expect(
      journeyConnections(graph, stage.id, recorded).outgoing[0],
    ).toMatchObject({ nodeId: "op-2", tensorId: "b", operandRole: "Input 1" });
    expect(journeyConnections(graph, "op-0", recorded)).toEqual({
      incoming: [],
      outgoing: [],
    });
  });
});

describe("returned tensor ownership", () => {
  it("attributes reused output IDs only to their latest recorded producer", () => {
    const recorded = trace(
      [op(0, ["x"], ["a", "b"]), op(1, ["b"], ["b"])],
      ["b", "a", "b"],
    );
    const graph = buildJourney(recorded);
    expect(graph.nodes.find((node) => node.id === "op-0")?.terminal).toBe(true);
    expect(returnedTensorIds(graph, "op-0", recorded)).toEqual(["a"]);
    expect(returnedTensorIds(graph, "op-1", recorded)).toEqual(["b"]);
    expect(returnedTensorIds(graph, "missing", recorded)).toEqual([]);
  });

  it("preserves trace return order and removes duplicate returned identities", () => {
    const recorded = trace(
      [op(0, ["x"], ["a", "b", "c"])],
      ["c", "a", "c", "b"],
    );
    expect(returnedTensorIds(buildJourney(recorded), "op-0", recorded)).toEqual(
      ["c", "a", "b"],
    );
  });

  it("attributes final states to their visible collapsed stage", () => {
    const recorded = trace(
      [op(0, ["x"], ["a", "b"]), op(1, ["a"], ["c"]), op(2, ["b"], ["b"])],
      ["a", "b", "c"],
    );
    const stage: JourneyStage = {
      id: "stage-block",
      parent_id: null,
      parentStageId: null,
      path: "block",
      module_type: "Block",
      title: "Block",
      start_index: 0,
      end_index: 2,
      inputs: ["x"],
      outputs: ["c"],
      operationIds: ["op-0", "op-1"],
      failed: false,
    };
    const run: Run = {
      id: "run",
      project_id: "project",
      created_at: "today",
      project: blankProject("Stages"),
      trace: recorded,
    };
    const graph = collapseJourney(
      buildJourney(recorded),
      [stage],
      new Set([stage.id]),
      run,
    );
    expect(returnedTensorIds(graph, stage.id, recorded)).toEqual(["a", "c"]);
    expect(returnedTensorIds(graph, "op-2", recorded)).toEqual(["b"]);
  });

  it("recognizes input passthrough only when no operation replaces that state", () => {
    const passthrough = trace([], ["x", "x"]);
    expect(
      returnedTensorIds(buildJourney(passthrough), "input-x", passthrough),
    ).toEqual(["x"]);
    const replaced = trace([op(0, ["x"], ["x"])], ["x"]);
    const graph = buildJourney(replaced);
    expect(returnedTensorIds(graph, "input-x", replaced)).toEqual([]);
    expect(returnedTensorIds(graph, "op-0", replaced)).toEqual(["x"]);
  });

  it("includes returned mutation states even without explicit operation outputs", () => {
    const mutation = {
      ...op(0, ["x"], []),
      mutations: [{ before: "x", after: "updated", kind: "alias" as const }],
    };
    const recorded = trace([mutation], ["updated"]);
    expect(
      returnedTensorIds(buildJourney(recorded), mutation.id, recorded),
    ).toEqual(["updated"]);
  });
});

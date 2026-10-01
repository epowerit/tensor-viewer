import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import { buildJourney, NODE_HEIGHT, NODE_WIDTH } from "./graph";
import { projectOperationScene } from "./sceneGraph";
import { sceneView } from "./scene";

function tensor(id: string, role: Tensor["role"] = "intermediate"): Tensor {
  return {
    id,
    name: id,
    role,
    shape: [2, 2],
    axes: ["rows", "columns"],
    dtype: "float32",
    strides: [2, 1],
    storage_id: id,
    storage_offset: 0,
    contiguous: true,
    numel: 4,
    values: [0, 1, 2, 3],
    minimum: 0,
    maximum: 3,
  };
}
function op(
  id: string,
  index: number,
  inputs: string[],
  outputs: string[],
): Operation {
  return {
    id,
    index,
    kind: "example",
    function: "example",
    inputs,
    outputs,
    arguments: {},
    source: { line: index + 1, text: id },
    module: "Example",
    status: "ok",
    error: null,
    lesson: {
      title: "Example",
      summary: "Example",
      detail: "Example",
      category: "layout",
      interaction: "inspect",
      mapping: null,
      axis_order: null,
    },
  };
}
function trace(operations: Operation[], values: Tensor[]): Run["trace"] {
  return {
    schema_version: "1",
    operations,
    tensors: Object.fromEntries(values.map((value) => [value.id, value])),
    input_ids: values
      .filter((value) => value.role === "input")
      .map((value) => value.id),
    output_ids: operations.at(-1)?.outputs ?? [],
    error: null,
    stdout: "",
    duration_ms: 0,
  };
}

describe("active operation tensor actors", () => {
  it("shows every split result while retaining the canonical operation and trace", () => {
    const recorded = trace(
      [op("split", 0, ["x"], ["a", "b"])],
      [tensor("x", "input"), tensor("a"), tensor("b")],
    );
    const graph = buildJourney(recorded);
    const before = structuredClone(graph);
    const projected = projectOperationScene(graph, "split", recorded.tensors);
    const results = projected.nodes.filter(
      (node) => projected.actorRoles[node.id]?.side === "output",
    );
    expect(results.map((node) => node.tensors[0].id)).toEqual(["a", "b"]);
    expect(results[0].id).toBe("split");
    expect(results.map((node) => projected.actorOrigins[node.id])).toEqual([
      "split",
      "split",
    ]);
    expect(projected.edges.map((edge) => edge.target)).toEqual(
      results.map((node) => node.id),
    );
    expect(projected.sceneNodeIds.size).toBe(3);
    expect(graph).toEqual(before);
    expect(recorded.operations[0].id).toBe("split");
    for (const node of projected.nodes) {
      expect(node.x + NODE_WIDTH).toBeLessThanOrEqual(projected.width);
      expect(node.y + NODE_HEIGHT).toBeLessThanOrEqual(projected.height);
    }
  });

  it("connects a join to both outputs of the same producer without hiding siblings", () => {
    const recorded = trace(
      [
        op("split", 0, ["x"], ["a", "b", "c"]),
        op("join", 1, ["a", "b"], ["out"]),
        op("later", 2, ["c"], ["end"]),
      ],
      [
        tensor("x", "input"),
        ...["a", "b", "c", "out", "end"].map((id) => tensor(id)),
      ],
    );
    const graph = projectOperationScene(
      buildJourney(recorded),
      "join",
      recorded.tensors,
    );
    const incoming = graph.edges.filter((edge) => edge.target === "join");
    expect(
      incoming.map(
        (edge) =>
          graph.nodes.find((node) => node.id === edge.source)!.tensors[0].id,
      ),
    ).toEqual(["a", "b"]);
    expect(new Set(incoming.map((edge) => edge.source)).size).toBe(2);
    const sibling = graph.nodes.find((node) => node.tensors[0]?.id === "c")!;
    expect(graph.sceneNodeIds.has(sibling.id)).toBe(false);
    expect(graph.edges.find((edge) => edge.target === "later")!.source).toBe(
      sibling.id,
    );
    expect(graph.nodes.find((node) => node.id === "later")!.tensors).toEqual([
      recorded.tensors.end,
    ]);
  });

  it("adds actual parameters and retains repeated operand positions", () => {
    const recorded = trace(
      [op("join", 0, ["x", "w", "w", "x"], ["out"])],
      [tensor("x", "input"), tensor("w", "parameter"), tensor("out")],
    );
    const base = buildJourney(recorded);
    expect(base.nodes.some((node) => node.tensors[0]?.id === "w")).toBe(false);
    const graph = projectOperationScene(base, "join", recorded.tensors);
    const inputs = graph.edges
      .filter((edge) => edge.target === "join")
      .sort((a, b) => a.inputIndex - b.inputIndex);
    expect(inputs.map((edge) => edge.tensorId)).toEqual(["x", "w", "w", "x"]);
    expect(inputs.map((edge) => edge.inputIndex)).toEqual([0, 1, 2, 3]);
    expect(inputs[1].source).toBe(inputs[2].source);
    expect(inputs[0].source).toBe(inputs[3].source);
    expect(new Set(inputs.map((edge) => edge.id)).size).toBe(4);
    expect(
      graph.nodes.filter((node) => node.tensors[0]?.id === "w"),
    ).toHaveLength(1);
    expect(graph.sceneNodeIds.size).toBe(3);
    expect(base.nodes.some((node) => node.tensors[0]?.id === "w")).toBe(false);
  });

  it("keeps parameter-only operation geometry after the actual input", () => {
    const recorded = trace(
      [op("lookup", 0, ["w"], ["out"])],
      [tensor("w", "parameter"), tensor("out")],
    );
    const graph = projectOperationScene(
      buildJourney(recorded),
      "lookup",
      recorded.tensors,
    );
    const input = graph.nodes.find((node) => node.tensors[0]?.id === "w")!;
    const output = graph.nodes.find((node) => node.id === "lookup")!;
    expect(input.x + NODE_WIDTH).toBeLessThan(output.x);
    expect(output.depth).toBe(1);
  });

  it("places late-layer weights beside actual operands and keeps the whole scene visible", () => {
    const layers = Array.from({ length: 8 }, (_, index) =>
      op(
        `layer${index}`,
        index,
        [index ? `value${index - 1}` : "x"],
        [`value${index}`],
      ),
    );
    const recorded = trace(
      [...layers, op("weighted", 8, ["value7", "weight"], ["out"])],
      [
        tensor("x", "input"),
        ...layers.map((_, index) => tensor(`value${index}`)),
        tensor("weight", "parameter"),
        tensor("out"),
      ],
    );
    const canonical = buildJourney(recorded);
    const graph = projectOperationScene(
      canonical,
      "weighted",
      recorded.tensors,
    );
    const preceding = graph.nodes.find((node) => node.id === "layer7")!;
    const weight = graph.nodes.find(
      (node) => node.tensors[0]?.id === "weight",
    )!;
    const output = graph.nodes.find((node) => node.id === "weighted")!;
    expect(weight.x).toBe(preceding.x);
    expect(weight.x + NODE_WIDTH).toBeLessThan(output.x);
    expect(graph.nodes.find((node) => node.id === "input-x")!.x).toBe(
      canonical.nodes.find((node) => node.id === "input-x")!.x,
    );
    const size = { width: 1000, height: 800 };
    const view = sceneView(graph, "weighted", size, { x: 0, y: 0, scale: 1 });
    for (const node of graph.nodes.filter((item) =>
      graph.sceneNodeIds.has(item.id),
    )) {
      expect(node.x * view.scale + view.x).toBeGreaterThanOrEqual(0);
      expect((node.x + NODE_WIDTH) * view.scale + view.x).toBeLessThanOrEqual(
        size.width,
      );
      expect(node.y * view.scale + view.y).toBeGreaterThanOrEqual(0);
      expect((node.y + NODE_HEIGHT) * view.scale + view.y).toBeLessThanOrEqual(
        size.height,
      );
    }
    expect(view.scale).toBeGreaterThan(0.55);
  });

  it("exposes writes and alias snapshots without converting storage edges to operands", () => {
    const write = op("write", 1, ["x"], []);
    write.mutations = [
      { before: "x", after: "x2", kind: "write" },
      { before: "alias", after: "alias2", kind: "alias" },
    ];
    const recorded = trace(
      [op("view", 0, ["x"], ["alias"]), write],
      [
        tensor("x", "input"),
        ...["x2", "alias", "alias2"].map((id) => tensor(id)),
      ],
    );
    const graph = projectOperationScene(
      buildJourney(recorded),
      "write",
      recorded.tensors,
    );
    const outputs = graph.nodes.filter(
      (node) => graph.actorRoles[node.id]?.side === "output",
    );
    expect(outputs.map((node) => node.tensors[0].id)).toEqual(["x2", "alias2"]);
    expect(graph.edges.filter((edge) => edge.kind === "storage")).toHaveLength(
      2,
    );
    expect(
      graph.edges
        .filter((edge) => edge.kind === "storage")
        .every((edge) => edge.tensorId === "alias"),
    ).toBe(true);
  });

  it("retains failed and tensor-free operations without inventing a result", () => {
    const failed = op("failed", 0, ["x"], []);
    failed.status = "error";
    const recorded = trace([failed], [tensor("x", "input")]);
    const graph = projectOperationScene(
      buildJourney(recorded),
      "failed",
      recorded.tensors,
    );
    expect(graph.nodes.find((node) => node.id === "failed")!.tensors).toEqual(
      [],
    );
    expect(graph.sceneNodeIds).toEqual(new Set(["failed", "input-x"]));
    expect(graph.edges).toHaveLength(1);
  });

  it("supports collapsed producer outputs without changing stage identity", () => {
    const recorded = trace(
      [
        op("producer", 0, ["x"], ["a", "b"]),
        op("join", 1, ["a", "b"], ["out"]),
      ],
      [tensor("x", "input"), ...["a", "b", "out"].map((id) => tensor(id))],
    );
    const base = buildJourney(recorded);
    const producer = base.nodes.find((node) => node.id === "producer")!;
    producer.id = "stage-call";
    producer.operation = undefined;
    base.edges = base.edges.map((edge) => ({
      ...edge,
      source: edge.source === "producer" ? "stage-call" : edge.source,
      target: edge.target === "producer" ? "stage-call" : edge.target,
    }));
    const graph = projectOperationScene(base, "join", recorded.tensors);
    const operands = graph.edges
      .filter((edge) => edge.target === "join")
      .map((edge) => edge.source);
    expect(operands.map((id) => graph.actorOrigins[id])).toEqual([
      "stage-call",
      "stage-call",
    ]);
    expect(new Set(operands).size).toBe(2);
  });

  it("uses distinct identifiers even when a model already has the generated-looking name", () => {
    const recorded = trace(
      [
        op("split", 0, ["x"], ["a", "b"]),
        op("scene:split:1", 1, ["b"], ["out"]),
      ],
      [tensor("x", "input"), ...["a", "b", "out"].map((id) => tensor(id))],
    );
    const graph = projectOperationScene(
      buildJourney(recorded),
      "split",
      recorded.tensors,
    );
    expect(new Set(graph.nodes.map((node) => node.id)).size).toBe(
      graph.nodes.length,
    );
    expect(graph.actorOrigins["scene:split:1"]).toBe("scene:split:1");
    expect(
      graph.nodes.find((node) => node.tensors[0]?.id === "b")!.id,
    ).not.toBe("scene:split:1");
  });

  it("preserves all output metadata at high fan-out without generating cells", () => {
    const outputs = Array.from({ length: 64 }, (_, index) => `output${index}`);
    const recorded = trace(
      [op("split", 0, ["x"], outputs)],
      [tensor("x", "input"), ...outputs.map((id) => tensor(id))],
    );
    const graph = projectOperationScene(
      buildJourney(recorded),
      "split",
      recorded.tensors,
    );
    expect(graph.sceneNodeIds.size).toBe(65);
    expect(graph.edges).toHaveLength(64);
    expect(
      graph.nodes
        .filter((node) => graph.actorRoles[node.id]?.side === "output")
        .flatMap((node) => node.tensors.map((value) => value.id)),
    ).toEqual(outputs);
  });

  it("bundles large fan-in and fan-out before allocating a Cartesian edge set", () => {
    const inputs = Array.from({ length: 40 }, (_, index) => `input${index}`);
    const outputs = Array.from({ length: 40 }, (_, index) => `output${index}`);
    const recorded = trace(
      [op("many", 0, inputs, outputs)],
      [
        ...inputs.map((id) => tensor(id, "input")),
        ...outputs.map((id) => tensor(id)),
      ],
    );
    const canonical = buildJourney(recorded);
    const graph = projectOperationScene(canonical, "many", recorded.tensors);
    const junction = graph.nodes.find((node) => node.junction)!;
    expect(graph.nodes.filter((node) => node.junction)).toHaveLength(1);
    expect(junction.tensors).toEqual([]);
    expect(junction.terminal).toBe(false);
    expect(junction.operation).toBe(recorded.operations[0]);
    expect(graph.actorOrigins[junction.id]).toBe("many");
    expect(graph.actorRoles[junction.id]).toBeUndefined();
    expect(graph.sceneNodeIds.size).toBe(81);
    expect(graph.edges).toHaveLength(inputs.length + outputs.length);
    expect(
      graph.edges
        .filter((edge) => edge.target === junction.id)
        .map((edge) => edge.tensorId),
    ).toEqual(inputs);
    expect(
      graph.edges
        .filter((edge) => edge.source === junction.id)
        .map((edge) => edge.tensorId),
    ).toEqual(outputs);
    expect(
      graph.nodes
        .filter((node) => graph.actorRoles[node.id]?.side === "output")
        .flatMap((node) => node.tensors.map((value) => value.id)),
    ).toEqual(outputs);
    expect(graph.nodes.find((node) => node.id === "many")!.tensors[0].id).toBe(
      outputs[0],
    );
    expect(canonical.nodes.some((node) => node.junction)).toBe(false);
    for (const node of graph.nodes) {
      expect(node.x + NODE_WIDTH).toBeLessThanOrEqual(graph.width);
      expect(node.y + NODE_HEIGHT).toBeLessThanOrEqual(graph.height);
    }
  });

  it("bundles a collapsed producer without moving the active operation's dependencies to the junction", () => {
    const inputs = Array.from({ length: 40 }, (_, index) => `input${index}`);
    const parts = Array.from({ length: 40 }, (_, index) => `part${index}`);
    const recorded = trace(
      [
        op("producer", 0, inputs, parts),
        op("consume", 1, [parts[7], parts[35]], ["out"]),
      ],
      [
        ...inputs.map((id) => tensor(id, "input")),
        ...[...parts, "out"].map((id) => tensor(id)),
      ],
    );
    const canonical = buildJourney(recorded);
    const producer = canonical.nodes.find((node) => node.id === "producer")!;
    producer.operation = undefined;
    producer.stage = {
      id: "stage-producer",
      parent_id: null,
      title: "Producer stage",
      path: "model.producer",
      module_type: "Producer",
      start_index: 0,
      end_index: 1,
      inputs,
      outputs: parts,
      operationIds: ["producer"],
      parentStageId: null,
      failed: false,
    };
    const graph = projectOperationScene(canonical, "consume", recorded.tensors);
    const junction = graph.nodes.find((node) => node.junction)!;
    expect(junction.stage).toBe(producer.stage);
    expect(junction.operation).toBeUndefined();
    expect(graph.edges).toHaveLength(inputs.length + parts.length + 2);
    expect(graph.sceneNodeIds.has(junction.id)).toBe(false);
    const actual = graph.edges.filter((edge) => edge.target === "consume");
    expect(
      actual.map(
        (edge) =>
          graph.nodes.find((node) => node.id === edge.source)!.tensors[0].id,
      ),
    ).toEqual([parts[7], parts[35]]);
    expect(actual.every((edge) => edge.source !== junction.id)).toBe(true);
    expect(graph.sceneNodeIds.size).toBe(3);
  });

  it("keeps late parameters before the junction and repeated operand positions intact", () => {
    const layers = Array.from({ length: 4 }, (_, index) =>
      op(
        `layer${index}`,
        index,
        [index ? `value${index - 1}` : "x"],
        [`value${index}`],
      ),
    );
    const outputs = Array.from({ length: 65 }, (_, index) => `output${index}`);
    const recorded = trace(
      [
        ...layers,
        op("many", 4, ["value3", "weight", "weight", "weight"], outputs),
      ],
      [
        tensor("x", "input"),
        ...layers.map((_, index) => tensor(`value${index}`)),
        tensor("weight", "parameter"),
        ...outputs.map((id) => tensor(id)),
      ],
    );
    const graph = projectOperationScene(
      buildJourney(recorded),
      "many",
      recorded.tensors,
    );
    const junction = graph.nodes.find((node) => node.junction)!;
    const weights = graph.edges.filter(
      (edge) => edge.target === junction.id && edge.tensorId === "weight",
    );
    expect(weights.map((edge) => edge.inputIndex)).toEqual([1, 2, 3]);
    expect(new Set(weights.map((edge) => edge.source)).size).toBe(1);
    const weight = graph.nodes.find((node) => node.id === weights[0].source)!;
    expect(weight.depth).toBe(junction.depth - 1);
    expect(weight.x).toBe(graph.nodes.find((node) => node.id === "layer3")!.x);
    expect(weight.x + NODE_WIDTH).toBeLessThan(junction.x);
    expect(graph.edges).toHaveLength(4 + 4 + outputs.length);
  });
});

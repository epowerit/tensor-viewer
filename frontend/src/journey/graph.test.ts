import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import { ancestors, buildJourney, NODE_HEIGHT, NODE_WIDTH } from "./graph";

function tensor(id: string, role: Tensor["role"] = "intermediate"): Tensor {
  return {
    id,
    name: id,
    role,
    shape: [1, 2, 2],
    axes: ["batch", "tokens", "features"],
    dtype: "float32",
    strides: [4, 2, 1],
    storage_id: id,
    storage_offset: 0,
    contiguous: true,
    numel: 4,
    values: [0, 1, 2, 3],
    minimum: 0,
    maximum: 3,
  };
}
function operation(
  id: string,
  index: number,
  inputs: string[],
  outputs: string[],
): Operation {
  return {
    id,
    index,
    kind: "reshape",
    function: "torch.reshape",
    inputs,
    outputs,
    arguments: {},
    source: { line: 50 - index, text: "x.reshape(...)" },
    module: "Example",
    status: "ok",
    error: null,
    lesson: {
      title: "Reshape",
      summary: "Reshape",
      detail: "Reshape",
      category: "layout",
      interaction: "inspect",
      mapping: null,
      axis_order: null,
    },
  };
}
function trace(
  operations: Operation[],
  tensors: Tensor[],
  output_ids: string[] = [],
): Run["trace"] {
  return {
    schema_version: "1",
    operations,
    tensors: Object.fromEntries(tensors.map((t) => [t.id, t])),
    input_ids: tensors.filter((t) => t.role === "input").map((t) => t.id),
    output_ids,
    error: null,
    stdout: "",
    duration_ms: 0,
  };
}

describe("tensor journey dataflow", () => {
  it("preserves forks and joins rather than chaining execution order", () => {
    const graph = buildJourney(
      trace(
        [
          operation("q", 0, ["x", "w"], ["qt"]),
          operation("k", 1, ["x", "w"], ["kt"]),
          operation("v", 2, ["x", "w"], ["vt"]),
          operation("scores", 3, ["qt", "kt"], ["st"]),
          operation("attend", 4, ["st", "vt"], ["out"]),
        ],
        [
          tensor("x", "input"),
          tensor("w", "parameter"),
          ...["qt", "kt", "vt", "st", "out"].map((id) => tensor(id)),
        ],
        ["out"],
      ),
    );
    expect(graph.nodes).toHaveLength(6);
    expect(graph.edges.map((e) => [e.source, e.target])).toEqual([
      ["input-x", "q"],
      ["input-x", "k"],
      ["input-x", "v"],
      ["q", "scores"],
      ["k", "scores"],
      ["scores", "attend"],
      ["v", "attend"],
    ]);
    expect(ancestors(graph, "scores")).toEqual(
      new Set(["scores", "q", "k", "input-x"]),
    );
    expect(graph.nodes.find((n) => n.id === "q")?.parameterCount).toBe(1);
    expect(graph.nodes.filter((n) => n.terminal).map((n) => n.id)).toEqual([
      "attend",
    ]);
    const branches = graph.nodes.filter((n) => ["q", "k", "v"].includes(n.id));
    expect(new Set(branches.map((n) => n.x)).size).toBe(1);
    expect(branches[1].y - branches[0].y).toBeGreaterThan(NODE_HEIGHT);
    expect(branches[2].y - branches[1].y).toBeGreaterThan(NODE_HEIGHT);
    for (const edge of graph.edges) {
      const a = graph.nodes.find((n) => n.id === edge.source)!;
      const b = graph.nodes.find((n) => n.id === edge.target)!;
      expect(a.x + NODE_WIDTH).toBeLessThan(b.x);
    }
  });

  it("retains both operand positions for x @ x", () => {
    const graph = buildJourney(
      trace(
        [operation("square", 0, ["x", "x"], ["y"])],
        [tensor("x", "input"), tensor("y")],
        ["y"],
      ),
    );
    expect(graph.edges.map((e) => e.inputIndex)).toEqual([0, 1]);
    expect(new Set(graph.edges.map((e) => e.id)).size).toBe(2);
  });

  it("connects all outputs of a tuple-producing operation to their consumers", () => {
    const graph = buildJourney(
      trace(
        [
          operation("split", 0, ["x"], ["a", "b"]),
          operation("left", 1, ["a"], ["c"]),
          operation("right", 2, ["b"], ["d"]),
        ],
        [tensor("x", "input"), ...["a", "b", "c", "d"].map((id) => tensor(id))],
        ["c", "d"],
      ),
    );
    expect(
      graph.nodes.find((n) => n.id === "split")?.tensors.map((t) => t.id),
    ).toEqual(["a", "b"]);
    expect(
      graph.edges.filter((e) => e.source === "split").map((e) => e.tensorId),
    ).toEqual(["a", "b"]);
    expect(graph.nodes.filter((n) => n.terminal).map((n) => n.id)).toEqual([
      "left",
      "right",
    ]);
  });

  it("keeps a failing operation and its input without inventing an output", () => {
    const failed = {
      ...operation("bad-view", 0, ["x"], []),
      status: "error" as const,
      error: "invalid shape",
    };
    const graph = buildJourney(trace([failed], [tensor("x", "input")]));
    expect(graph.nodes.find((n) => n.id === "bad-view")?.tensors).toEqual([]);
    expect(graph.edges[0].source).toBe("input-x");
    expect(graph.nodes.some((n) => n.terminal)).toBe(false);
  });

  it("handles unchanged inputs, factories, and unproduced constants", () => {
    const unchanged = buildJourney(trace([], [tensor("x", "input")], ["x"]));
    expect(unchanged.nodes[0].terminal).toBe(true);
    expect(unchanged.edges).toEqual([]);
    const factory = buildJourney(
      trace(
        [
          operation("create", 0, [], ["a"]),
          operation("add", 1, ["a", "constant"], ["b"]),
        ],
        [tensor("a"), tensor("constant"), tensor("b")],
        ["b"],
      ),
    );
    expect(factory.nodes.map((n) => n.id)).toContain("input-constant");
    expect(factory.edges.map((e) => e.source)).toEqual([
      "create",
      "input-constant",
    ]);
    expect(factory.width).toBeGreaterThan(0);
    expect(factory.height).toBeGreaterThan(0);
  });

  it("uses chronological producers when a tensor ID is reused", () => {
    const graph = buildJourney(
      trace(
        [
          operation("same", 0, ["x"], ["x"]),
          operation("next", 1, ["x"], ["y"]),
        ],
        [tensor("x", "input"), tensor("y")],
        ["y"],
      ),
    );
    expect(graph.edges.map((e) => [e.source, e.target])).toEqual([
      ["input-x", "same"],
      ["same", "next"],
    ]);
    expect(graph.edges.every((e) => e.source !== e.target)).toBe(true);
  });
});

describe("shared-storage provenance", () => {
  it("connects an updated alias to its view and the actual writer", () => {
    const view = operation("view", 0, ["x"], ["y"]);
    const write = operation("write", 1, ["x"], ["x2"]);
    write.mutations = [
      { before: "x", after: "x2", kind: "write" },
      { before: "y", after: "y2", kind: "alias" },
    ];
    const use = operation("use", 2, ["y2"], ["out"]);
    const graph = buildJourney(
      trace(
        [view, write, use],
        [
          tensor("x", "input"),
          ...["y", "x2", "y2", "out"].map((id) => tensor(id)),
        ],
        ["out"],
      ),
    );
    expect(graph.nodes.map((n) => n.id)).toEqual([
      "input-x",
      "view",
      "write",
      "use",
    ]);
    expect(graph.edges.map((e) => [e.source, e.target, e.kind])).toEqual([
      ["input-x", "view", "operand"],
      ["input-x", "write", "operand"],
      ["view", "write", "storage"],
      ["write", "use", "operand"],
    ]);
    expect(ancestors(graph, "use")).toEqual(
      new Set(["use", "write", "view", "input-x"]),
    );
    expect(
      graph.nodes.find((n) => n.id === "write")?.tensors.map((t) => t.id),
    ).toEqual(["x2", "y2"]);
  });
  it("shows side effects and final aliased returns even when an operation returns nothing", () => {
    const write = operation("assign", 0, ["x"], []);
    write.mutations = [{ before: "x", after: "x2", kind: "write" }];
    const graph = buildJourney(
      trace([write], [tensor("x", "input"), tensor("x2")], ["x2"]),
    );
    expect(graph.nodes.map((n) => n.id)).toEqual(["input-x", "assign"]);
    expect(graph.nodes[1].tensors[0].id).toBe("x2");
    expect(graph.nodes[1].terminal).toBe(true);
  });
  it("chains repeated mutations without cycles or rewriting earlier producers", () => {
    const first = operation("first", 0, ["x"], ["a"]);
    first.mutations = [{ before: "x", after: "a", kind: "write" }];
    const consume = operation("consume", 1, ["a"], ["b"]);
    const second = operation("second", 2, ["a"], ["c"]);
    second.mutations = [{ before: "a", after: "c", kind: "write" }];
    const graph = buildJourney(
      trace(
        [first, consume, second],
        [tensor("x", "input"), ...["a", "b", "c"].map((id) => tensor(id))],
        ["b", "c"],
      ),
    );
    expect(graph.edges.map((e) => [e.source, e.target])).toEqual([
      ["input-x", "first"],
      ["first", "consume"],
      ["first", "second"],
    ]);
    expect(
      graph.nodes.every(
        (n) => !graph.edges.some((e) => e.source === n.id && e.target === n.id),
      ),
    ).toBe(true);
  });
  it("does not infer mutations merely because two tensors share storage", () => {
    const x = tensor("x", "input"),
      y = tensor("y"),
      z = tensor("z");
    y.storage_id = x.storage_id;
    const graph = buildJourney(
      trace(
        [
          operation("old", 0, ["x"], ["z"]),
          operation("use", 1, ["y"], ["out"]),
        ],
        [x, y, z, tensor("out")],
      ),
    );
    expect(graph.edges.find((e) => e.target === "use")?.source).toBe("input-y");
  });
});

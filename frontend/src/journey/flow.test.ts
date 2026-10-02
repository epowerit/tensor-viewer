import { expect, test } from "vitest";
import type { JourneyEdge, JourneyNode } from "./graph";
import { edgeWidths, flowNeighbor, mainPath, spreadJumps } from "./flow";

const node = (id: string, numel: number): JourneyNode =>
  ({
    id,
    tensors: [{ id: `t-${id}`, numel, shape: [numel] }],
  }) as unknown as JourneyNode;
const edge = (source: string, target: string, inputIndex = 0): JourneyEdge => ({
  id: `${source}-${target}`,
  source,
  target,
  tensorId: `t-${source}`,
  inputIndex,
});
const graph = {
  nodes: [
    node("x", 4),
    node("mask", 1024),
    node("scores", 64),
    node("out", 16),
  ],
  edges: [
    edge("x", "scores"),
    edge("mask", "scores", 1),
    edge("scores", "out"),
  ],
};

test("connections carrying more data draw thicker, on an absolute log scale", () => {
  const widths = edgeWidths(graph);
  expect(widths.get("x-scores")).toBeCloseTo(1.2 + 0.5 * Math.log10(4));
  expect(widths.get("scores-out")).toBeCloseTo(1.2 + 0.5 * Math.log10(64));
  expect(widths.get("mask-scores")).toBeCloseTo(1.2 + 0.5 * Math.log10(1024));
  // Similar sizes look alike: 80 and 100 values differ by a twentieth of a pixel.
  const near = edgeWidths({
    nodes: [node("a", 80), node("b", 100), node("c", 1)],
    edges: [edge("a", "c"), edge("b", "c", 1)],
  });
  expect(near.get("b-c")! - near.get("a-c")!).toBeLessThan(0.06);
});

test("the main path follows each step's first operand back to the start", () => {
  expect(mainPath(graph, "out").map((n) => n.id)).toEqual([
    "x",
    "scores",
    "out",
  ]);
  expect(mainPath(graph, "x").map((n) => n.id)).toEqual(["x"]);
});

test("spread jumps flag exploding or vanishing values along a path", () => {
  const spread = (id: string, std?: number) =>
    ({
      id,
      tensors: [
        { id: `t-${id}`, histogram: std === undefined ? null : { std } },
      ],
    }) as unknown as JourneyNode;
  const jumps = spreadJumps([
    spread("a", 1),
    spread("b", 1.5),
    spread("c", 40),
    spread("shape-only"),
    spread("d", 2),
  ]);
  expect([...jumps.keys()]).toEqual(["c", "d"]);
  expect(jumps.get("c")).toBeCloseTo(40 / 1.5);
  expect(jumps.get("d")).toBeCloseTo(2 / 40);
});

test("flow neighbours: the earliest reader forward, the first operand back", () => {
  const ordered = {
    nodes: graph.nodes.map((n, i) => ({ ...n, operation: { index: i } })),
    edges: graph.edges,
  } as unknown as typeof graph;
  expect(flowNeighbor(ordered, "x", "forward")).toBe("scores");
  expect(flowNeighbor(ordered, "scores", "back")).toBe("x");
  expect(flowNeighbor(ordered, "out", "forward")).toBeUndefined();
});

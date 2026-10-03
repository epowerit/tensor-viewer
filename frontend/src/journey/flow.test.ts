import { expect, test } from "vitest";
import type { JourneyEdge, JourneyNode } from "./graph";
import {
  changeValues,
  edgeWidths,
  flowNeighbor,
  flowSibling,
  lensColor,
  lensScale,
  lensText,
  lensValue,
  mainPath,
  routeEdge,
  spreadJumps,
} from "./flow";
import { NODE_HEIGHT, NODE_WIDTH } from "./graph";

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

test("a connection that would cross nodes arcs over them", () => {
  const column = (i: number, y = 0) => ({ id: `n${i}`, x: i * 282, y });
  const from = column(0),
    to = column(3);
  const middle = [column(1), column(2)];
  const x = NODE_WIDTH,
    y = NODE_HEIGHT / 2;
  const skip = routeEdge(from, to, x, y, y, [from, ...middle, to]);
  expect(skip.skip).toBe(true);
  // It runs level just above the nodes it skips.
  expect(skip.path).toContain(`,${-26} L`);
  // Neighbours, and nodes off the line, keep the plain curve.
  expect(routeEdge(from, column(1), x, y, y, [from, column(1)]).skip).toBe(
    false,
  );
  expect(
    routeEdge(from, to, x, y, y, [from, column(1, 600), column(2, 600), to])
      .skip,
  ).toBe(false);
});

test("a flow lens reads a statistic and places it on a colour scale", () => {
  const tensor = {
    histogram: {
      low: -3,
      high: 2,
      counts: [3, 1],
      zeros: 1,
      non_finite: 0,
      std: 0.5,
    },
  } as unknown as Parameters<typeof lensValue>[0];
  expect(lensValue(tensor, "spread")).toBe(0.5);
  expect(lensValue(tensor, "magnitude")).toBe(3);
  expect(lensValue(tensor, "zeros")).toBe(0.25);
  expect(lensValue(undefined, "spread")).toBeUndefined();
  const scale = lensScale([0.1, 1, 10], "spread");
  expect([scale(0.1), scale(1), scale(10)]).toEqual([0, 0.5, 1]);
  expect(lensScale([], "zeros")(0.4)).toBe(0.4);
  expect(lensColor(0)).toBe("rgb(110, 168, 217)");
  expect(lensColor(1)).toBe("rgb(242, 166, 108)");
  expect(lensText(0.25, "zeros")).toBe("25% zeros");
  expect(lensText(0.94, "spread")).toBe("σ 0.94");
});

test("the change lens maps a run comparison onto the steps", () => {
  const diff = (id: string, change: string, delta: number | null = null) =>
    ({ left: { id }, change, delta }) as unknown as Parameters<
      typeof changeValues
    >[0][number];
  const values = changeValues([
    diff("op0", "same"),
    diff("op1", "values", 0.5),
    diff("op2", "shape"),
  ]);
  expect([...values.values()]).toEqual([0, 0.5, Infinity]);
  const scale = lensScale([...values.values()], "change");
  expect(scale(0)).toBe(0);
  expect(scale(Infinity)).toBe(1);
  expect(scale(0.5)).toBeGreaterThan(0.2);
  expect(lensText(0, "change")).toBe("unchanged");
  expect(lensText(Infinity, "change")).toBe("shape changed");
  expect(lensText(0.5, "change")).toBe("Δ 0.5");
});

test("siblings are the other readers of the same tensor, in step order", () => {
  const fan = {
    nodes: ["x", "q", "k", "v"].map((id, index) => ({
      ...node(id, 4),
      operation: { index },
    })),
    edges: [edge("x", "q"), edge("x", "k"), edge("x", "v")],
  } as unknown as typeof graph;
  expect(flowSibling(fan, "q", "next")).toBe("k");
  expect(flowSibling(fan, "v", "next")).toBe("q");
  expect(flowSibling(fan, "q", "previous")).toBe("v");
  expect(flowSibling(graph, "out", "next")).toBeUndefined();
});

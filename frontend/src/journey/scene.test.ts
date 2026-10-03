import { describe, expect, it } from "vitest";
import {
  NODE_HEIGHT,
  NODE_WIDTH,
  type JourneyGraph,
  type JourneyNode,
} from "./graph";
import { operationScene, sceneView } from "./scene";

const node = (id: string, x: number, y: number): JourneyNode => ({
  id,
  x,
  y,
  depth: 0,
  tensors: [],
  parameterCount: 0,
  terminal: false,
});
const graph: JourneyGraph = {
  nodes: [
    node("q", 0, 0),
    node("k", 0, 288),
    node("scores", 282, 144),
    node("probabilities", 564, 144),
  ],
  edges: [
    {
      id: "q-scores",
      source: "q",
      target: "scores",
      tensorId: "t1",
      inputIndex: 0,
    },
    {
      id: "k-scores",
      source: "k",
      target: "scores",
      tensorId: "t2",
      inputIndex: 1,
    },
    {
      id: "scores-probabilities",
      source: "scores",
      target: "probabilities",
      tensorId: "t3",
      inputIndex: 0,
    },
  ],
  width: 748,
  height: 480,
};
const size = { width: 1100, height: 800 };
const previous = { x: 0, y: 0, scale: 0.1 };

describe("recorded operation scenes", () => {
  it("includes every actual incoming dependency, without future computations", () => {
    const scene = operationScene(graph, "scores");
    expect([...scene.nodes]).toEqual(["scores", "q", "k"]);
    expect([...scene.edges]).toEqual(["q-scores", "k-scores"]);
    expect(operationScene(graph, "missing").nodes.size).toBe(0);
  });

  it("frames a branched operation with both operands and result visible", () => {
    const view = sceneView(graph, "scores", size, previous);
    for (const id of operationScene(graph, "scores").nodes) {
      const actor = graph.nodes.find((item) => item.id === id)!;
      expect(actor.x * view.scale + view.x).toBeGreaterThan(0);
      expect(actor.y * view.scale + view.y).toBeGreaterThan(0);
      expect(actor.y * view.scale + view.y).toBeGreaterThanOrEqual(128);
      expect((actor.x + NODE_WIDTH) * view.scale + view.x).toBeLessThan(
        size.width,
      );
      expect((actor.y + NODE_HEIGHT) * view.scale + view.y).toBeLessThan(
        size.height,
      );
      expect((actor.y + NODE_HEIGHT) * view.scale + view.y).toBeLessThanOrEqual(
        size.height - 96,
      );
    }
    expect(view.scale).toBeGreaterThanOrEqual(0.55);
  });

  it("keeps a high fan-in readable and the result centered", () => {
    const large = {
      ...graph,
      nodes: graph.nodes.map((item) =>
        item.id === "q" ? { ...item, y: -9000 } : item,
      ),
    };
    const view = sceneView(large, "scores", size, previous);
    const active = large.nodes.find((item) => item.id === "scores")!;
    expect(view.scale).toBe(0.45);
    expect((active.x + NODE_WIDTH / 2) * view.scale + view.x).toBe(
      size.width / 2,
    );
    expect(operationScene(large, "scores").edges.size).toBe(2);
  });

  it("fits an operand and its result inside a narrow canvas", () => {
    const narrow = { width: 340, height: 719 };
    const view = sceneView(graph, "probabilities", narrow, previous);
    for (const id of operationScene(graph, "probabilities").nodes) {
      const actor = graph.nodes.find((item) => item.id === id)!;
      expect(actor.x * view.scale + view.x).toBeGreaterThanOrEqual(0);
      expect((actor.x + NODE_WIDTH) * view.scale + view.x).toBeLessThanOrEqual(
        narrow.width,
      );
    }
    expect(view.scale).toBeGreaterThanOrEqual(0.45);
  });

  it("shrinks a short canvas's scene below the caption, cells still drawn", () => {
    const short = { width: 1200, height: 490 };
    const view = sceneView(graph, "scores", short, previous);
    expect(view.scale).toBeGreaterThanOrEqual(0.45);
    expect(view.scale).toBeLessThan(0.55);
    for (const id of operationScene(graph, "scores").nodes) {
      const actor = graph.nodes.find((item) => item.id === id)!;
      expect(actor.y * view.scale + view.y).toBeGreaterThanOrEqual(148);
    }
  });

  it("retains camera when selected tensor disappeared or canvas is hidden", () => {
    expect(sceneView(graph, "unknown", size, previous)).toBe(previous);
    expect(sceneView(graph, "scores", { width: 0, height: 0 }, previous)).toBe(
      previous,
    );
  });
});

describe("tall scenes", () => {
  // A step with three operands stacked high above it, as a layer norm with
  // its input, weight, and bias.
  const tall: JourneyGraph = {
    nodes: [
      node("x", 0, 0),
      node("weight", 0, 240),
      node("bias", 0, 480),
      node("norm", 282, 400),
    ],
    edges: ["x", "weight", "bias"].map((source, inputIndex) => ({
      id: `${source}-norm`,
      source,
      target: "norm",
      tensorId: `t-${source}`,
      inputIndex,
    })),
    width: 466,
    height: 672,
  };
  const short = { width: 1200, height: 520 };
  it("hang from the caption, keeping the active step whole", () => {
    const view = sceneView(tall, "norm", short, previous);
    expect(view.scale).toBe(0.45);
    // The topmost operand starts at the caption's edge, not under it.
    expect(0 * view.scale + view.y).toBeGreaterThanOrEqual(148);
    const norm = tall.nodes.find((item) => item.id === "norm")!;
    expect((norm.y + NODE_HEIGHT) * view.scale + view.y).toBeLessThanOrEqual(
      short.height,
    );
  });
});

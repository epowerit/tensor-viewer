import { expect, it } from "vitest";
import { NODE_HEIGHT, type JourneyGraph } from "./graph";
import { edgeDescription, journeyPorts, portPosition } from "./ports";

it("keeps every incoming operand attached to a large join", () => {
  for (const count of [1, 2, 32, 128, 512]) {
    const positions = Array.from({ length: count }, (_, i) =>
      portPosition(i, count),
    );
    expect(Math.min(...positions)).toBeGreaterThan(0);
    expect(Math.max(...positions)).toBeLessThan(NODE_HEIGHT);
    expect(new Set(positions).size).toBe(count);
  }
});

const graph = () =>
  ({
    nodes: [
      {
        id: "split",
        tensors: [
          { id: "a", name: "left", shape: [2, 3] },
          { id: "b", name: "right", shape: [2, 5] },
        ],
      },
      { id: "join", tensors: [], operation: { kind: "cat" } },
    ],
    edges: [
      {
        id: "e0",
        source: "split",
        target: "join",
        tensorId: "a",
        inputIndex: 0,
        kind: "operand",
      },
      {
        id: "e1",
        source: "split",
        target: "join",
        tensorId: "b",
        inputIndex: 127,
        kind: "operand",
      },
      {
        id: "e2",
        source: "split",
        target: "join",
        tensorId: "a",
        inputIndex: 128,
        kind: "storage",
      },
    ],
  }) as unknown as JourneyGraph;

it("routes tuple outputs separately and repeated uses from the same output port", () => {
  const ports = journeyPorts(graph());
  expect(ports.get("e0")!.sourceY).not.toBe(ports.get("e1")!.sourceY);
  expect(ports.get("e0")!.sourceY).toBe(ports.get("e2")!.sourceY);
  expect(new Set([...ports.values()].map((port) => port.targetY)).size).toBe(3);
  for (const port of ports.values()) {
    expect(port.targetY).toBeGreaterThan(0);
    expect(port.targetY).toBeLessThan(NODE_HEIGHT);
  }
});

it("describes the actual carried tensor and operand or storage role", () => {
  const full = graph();
  expect(edgeDescription(full, full.edges[1])).toBe(
    "right [2, 5] → cat, input 128",
  );
  expect(edgeDescription(full, full.edges[2])).toBe(
    "left [2, 3] → cat, shared storage",
  );
  full.nodes[1].stage = { title: "Block" } as never;
  expect(edgeDescription(full, full.edges[0])).toBe(
    "left [2, 3] → Block, used within stage",
  );
});

it("bundles connections at the center of a compact operation junction", () => {
  const full = graph();
  full.nodes[0].junction = true;
  full.nodes[1].junction = true;
  for (const port of journeyPorts(full).values()) {
    expect(port.sourceY).toBe(NODE_HEIGHT / 2);
    expect(port.targetY).toBe(NODE_HEIGHT / 2);
  }
});

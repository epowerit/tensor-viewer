import { describe, it, expect } from "vitest";
import type { ComponentSpec } from "../api/client";
import { componentSources, hasBranches } from "./connections";
const nodes: ComponentSpec[] = [
  { id: "a", kind: "linear", parameters: {} },
  { id: "b", kind: "relu", parameters: {} },
  { id: "c", kind: "add_join", parameters: {} },
];
describe("builder graph connections", () => {
  it("preserves sequential defaults and makes residual joins explicit", () => {
    expect(componentSources(nodes, 0)).toEqual(["input"]);
    expect(componentSources(nodes, 1)).toEqual(["a"]);
    expect(componentSources(nodes, 2)).toEqual(["b", "input"]);
    expect(hasBranches(nodes)).toBe(true);
    expect(hasBranches(nodes.slice(0, 2))).toBe(false);
  });
  it("gives every two-input join the same source defaults", () => {
    for (const kind of ["add_join", "concat_join", "stack_join"]) {
      const join = { id: "join", kind, parameters: {} };
      expect(componentSources([join], 0)).toEqual(["input", "input"]);
      expect(componentSources([...nodes.slice(0, 2), join], 2)).toEqual([
        "b",
        "input",
      ]);
      expect(hasBranches([join])).toBe(true);
    }
  });
  it("retains explicit references when a dependency moves or is removed", () => {
    const branch = [nodes[0], { ...nodes[1], sources: ["input"] }];
    expect(hasBranches(branch)).toBe(true);
    expect(componentSources([{ ...nodes[1], sources: ["a"] }], 0)).toEqual([
      "a",
    ]);
  });
});

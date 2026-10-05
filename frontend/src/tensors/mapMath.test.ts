import { expect, test } from "vitest";
import type { Run } from "../api/client";
import { layerStates } from "./layerStates";
import { readNpy } from "./npy";
import { project2d } from "./pca";

// Written by numpy.save.
const F32 =
  "k05VTVBZAQB2AHsnZGVzY3InOiAnPGY0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDMpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoAAAAAAAAAPwAAgD8AAMA/AAAAQAAAIEA=";
const I64 =
  "k05VTVBZAQB2AHsnZGVzY3InOiAnPGk4JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIsIDIpLCB9ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAoBAAAAAAAAAP7/////////AwAAAAAAAAAAAAAAAAEAAA==";
const bytes = (text: string) =>
  Uint8Array.from(atob(text), (c) => c.charCodeAt(0)).buffer;

test("a .npy file reads as its shape and values", () => {
  const floats = readNpy(bytes(F32));
  expect(floats?.shape).toEqual([2, 3]);
  expect([...floats!.values]).toEqual([0, 0.5, 1, 1.5, 2, 2.5]);
  const ints = readNpy(bytes(I64));
  expect([...ints!.values]).toEqual([1, -2, 3, 2 ** 40]);
  expect(readNpy(new ArrayBuffer(4))).toBeNull();
});

test("rows spread along one line project onto the first direction", () => {
  const rows = [-2, -1, 0, 1, 2].map((t) => [t, 2 * t, 0.001 * t * t]);
  const found = project2d(rows)!;
  expect(found.explained[0]).toBeGreaterThan(0.999);
  const xs = found.points.map(([x]) => x);
  // Evenly spaced along the line, |step| = √5, in order.
  for (let i = 1; i < xs.length; i++)
    expect(Math.abs(xs[i] - xs[i - 1])).toBeCloseTo(Math.sqrt(5), 3);
  expect(found.points.every(([, y]) => Math.abs(y) < 0.01)).toBe(true);
});

test("a plane's two directions split its spread", () => {
  const rows = [
    [3, 0, 0],
    [-3, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
  ];
  const found = project2d(rows)!;
  expect(found.explained[0]).toBeCloseTo(0.9, 6);
  expect(found.explained[1]).toBeCloseTo(0.1, 6);
  expect(project2d([[1, 2]])).toBeNull();
});

test("the block stack's states are found as the backend finds them", () => {
  const call = (
    id: string,
    path: string,
    type: string,
    parent: string | null,
  ) => ({
    id,
    parent_id: parent,
    path,
    module_type: type,
    start_index: 0,
    end_index: 1,
    inputs: [`${id}-in`],
    outputs: [`${id}-out`],
  });
  const trace = {
    module_calls: [
      call("root", "GPT", "GPT", null),
      call("e", "embed", "Embedding", "root"),
      call("b0", "blocks.0", "Block", "root"),
      call("b1", "blocks.1", "Block", "root"),
      call("n", "norm", "LayerNorm", "root"),
    ],
  } as unknown as Run["trace"];
  expect(layerStates(trace)).toEqual([
    { name: "before blocks.0", tensorId: "b0-in" },
    { name: "blocks.0", tensorId: "b0-out" },
    { name: "blocks.1", tensorId: "b1-out" },
  ]);
});

test("similarity is the cosine of each pair of rows", async () => {
  const { similarity } = await import("../shell/MapPanel");
  const found = similarity([
    [1, 0],
    [2, 0],
    [0, 3],
    [-1, 0],
    [0, 0],
  ]);
  expect(found[0][1]).toBeCloseTo(1);
  expect(found[0][2]).toBeCloseTo(0);
  expect(found[0][3]).toBeCloseTo(-1);
  // A zero row is like nothing.
  expect(found[4]).toEqual([0, 0, 0, 0, 0]);
});

import { expect, test } from "vitest";
import type { Run } from "../api/client";
import {
  filterFlow,
  flowCsv,
  flowRows,
  moduleTree,
  sortFlow,
} from "./flowRows";

const histogram = (std: number, zeros: number, low: number, high: number) => ({
  std,
  zeros,
  low,
  high,
  counts: [4],
  non_finite: 0,
});
const trace = {
  operations: [
    {
      id: "op0",
      index: 0,
      kind: "linear",
      outputs: ["t0"],
      status: "ok",
      source: { line: 3 },
    },
    {
      id: "op1",
      index: 1,
      kind: "relu",
      inputs: ["t0"],
      outputs: ["t1"],
      status: "ok",
      source: { line: 4 },
      module: "Net / attention",
    },
    { id: "op2", index: 2, kind: "__getitem__", outputs: ["t2"], status: "ok" },
  ],
  tensors: {
    t0: {
      name: "h",
      shape: [2, 8],
      dtype: "float32",
      numel: 16,
      histogram: histogram(1.5, 0, -4, 3),
    },
    t1: {
      name: "a",
      shape: [2, 8],
      dtype: "float32",
      numel: 16,
      histogram: histogram(0.4, 2, 0, 3),
    },
    t2: {
      name: "a[0]",
      shape: [8],
      dtype: "float32",
      numel: 8,
      histogram: null,
    },
  },
} as unknown as Run["trace"];

test("each step becomes a row with its shape and value statistics", () => {
  const rows = flowRows(trace);
  expect(rows.map((row) => [row.step, row.operation, row.name])).toEqual([
    [1, "linear", "h"],
    [2, "relu", "a"],
    [3, "index", "a[0]"],
  ]);
  expect(rows[1].zeros).toBe(0.5);
  expect(rows[2].spread).toBeNull();
});

test("measures sort largest first, with missing values last", () => {
  const rows = flowRows(trace);
  expect(sortFlow(rows, "spread").map((row) => row.step)).toEqual([1, 2, 3]);
  expect(sortFlow(rows, "zeros").map((row) => row.step)).toEqual([2, 1, 3]);
  expect(sortFlow(rows, "magnitude").map((row) => row.step)).toEqual([1, 2, 3]);
  expect(sortFlow(rows, "size").map((row) => row.step)).toEqual([1, 2, 3]);
  expect(
    sortFlow(sortFlow(rows, "zeros"), "step").map((row) => row.step),
  ).toEqual([1, 2, 3]);
});

test("rows name their inputs and filter by any word", () => {
  const rows = flowRows(trace);
  expect(rows[1].inputs).toEqual(["h"]);
  expect(filterFlow(rows, "attention").map((row) => row.step)).toEqual([2]);
  expect(filterFlow(rows, "RELU h").map((row) => row.step)).toEqual([2]);
  expect(filterFlow(rows, "  ")).toHaveLength(3);
});

test("the change column sorts structural changes first, then the largest", () => {
  const rows = flowRows(trace).map((row, i) => ({
    ...row,
    change: [0.5, Infinity, 0][i],
  }));
  expect(sortFlow(rows, "change").map((row) => row.step)).toEqual([2, 1, 3]);
});

test("rows group into the modules they ran in, with totals", () => {
  const rows = flowRows(trace).map((row, i) => ({
    ...row,
    module: ["Net", "Net / attention", ""][i],
  }));
  const tree = moduleTree(rows);
  expect(tree.map((group) => group.name)).toEqual(["(top level)", "Net"]);
  const net = tree[1];
  expect(net.steps).toBe(2);
  expect(net.values).toBe(32);
  expect(net.spread).toBe(1.5);
  // A one-step call reads as a row of its parent, not a group of its own.
  expect(net.children).toEqual([]);
  expect(net.rows.map((row) => row.module)).toEqual(["Net", "Net / attention"]);
});

test("the flow copies as CSV with quoted text where needed", () => {
  const csv = flowCsv(flowRows(trace));
  const [header, first] = csv.split("\n");
  expect(header.startsWith("step,operation,result,inputs,shape")).toBe(true);
  expect(first.startsWith("1,linear,h,,2x8,float32,16,1.5,0,-4,3,3")).toBe(
    true,
  );
  expect(
    flowCsv([{ ...flowRows(trace)[0], name: 'a, "b"' }]).split("\n")[1],
  ).toContain('"a, ""b"""');
});

test("nested module names drop their parent's prefix", () => {
  const rows = flowRows(trace).map((row) => ({
    ...row,
    module: "CLIP / encoder / encoder.block / encoder.block.norm1",
  }));
  const names: string[] = [];
  let groups = moduleTree(rows);
  while (groups.length) {
    names.push(groups[0].name);
    groups = groups[0].children;
  }
  expect(names).toEqual(["CLIP", "encoder", "block", "norm1"]);
});

test("steps holding NaN or infinity say so, and the filter finds them", () => {
  const faulty = {
    ...trace,
    tensors: {
      ...trace.tensors,
      t1: {
        ...trace.tensors.t1,
        histogram: { ...histogram(0.4, 2, 0, 3), non_finite: 3 },
      },
    },
  } as unknown as Run["trace"];
  const rows = flowRows(faulty);
  expect(rows.map((row) => row.broken)).toEqual([0, 3, 0]);
  expect(filterFlow(rows, "nan").map((row) => row.step)).toEqual([2]);
  expect(flowCsv(rows).split("\n")[2].endsWith(",3")).toBe(true);
});

test("steps with contracts are found by the word contract, broken ones by broken", () => {
  const rows = flowRows(trace).map((row, i) => ({
    ...row,
    contract: (["kept", "broken", undefined] as const)[i],
  }));
  expect(filterFlow(rows, "contract").map((row) => row.step)).toEqual([1, 2]);
  expect(filterFlow(rows, "broken").map((row) => row.step)).toEqual([2]);
});

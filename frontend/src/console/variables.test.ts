import { expect, test } from "vitest";
import type { Run } from "../api/client";
import { variables } from "./variables";

const tensor = (
  id: string,
  name: string,
  storage: string,
  role = "intermediate",
) => ({
  id,
  name,
  storage_id: storage,
  role,
  shape: [2, 3],
});

test("variables keep their latest state, producer, and storage sharing", () => {
  const trace = {
    input_ids: ["t0"],
    tensors: {
      t0: tensor("t0", "x", "s1", "input"),
      t1: tensor("t1", "view", "s1"),
      t2: tensor("t2", "frozen", "s2"),
      t3: tensor("t3", "x", "s1", "input"),
      t4: tensor("t4", "view", "s1"),
      t5: tensor("t5", "weight", "s3", "parameter"),
      t6: tensor("t6", "transpose", "s1"),
    },
    operations: [
      { id: "op0", kind: "view", outputs: ["t1"], source: { line: 1 } },
      { id: "op1", kind: "clone", outputs: ["t2"], source: { line: 2 } },
      {
        id: "op2",
        kind: "add_",
        outputs: ["t3"],
        mutations: [
          { before: "t0", after: "t3", kind: "write" },
          { before: "t1", after: "t4", kind: "write" },
        ],
        source: { line: 3 },
      },
      { id: "op3", kind: "linear", outputs: ["t5"], source: { line: 4 } },
      {
        id: "op4",
        kind: "transpose",
        outputs: ["t6"],
        source: { line: 5, file: "helper.py" },
      },
    ],
  } as unknown as Run["trace"];
  const result = variables(trace);
  expect(result.map((v) => v.name)).toEqual([
    "x",
    "view",
    "frozen",
    "transpose",
  ]);
  const [x, view, frozen, anonymous] = result;
  expect([x.tensor.id, x.nodeId, x.line, x.states]).toEqual([
    "t3",
    "op2",
    3,
    2,
  ]);
  expect([view.tensor.id, view.nodeId, view.states]).toEqual(["t4", "op2", 2]);
  expect(view.sharedWith).toEqual(["x", "transpose"]);
  expect(frozen.sharedWith).toEqual([]);
  // `view` is a user name that happens to match its operation.
  expect(anonymous.anonymous).toBe(true);
  expect(anonymous.line).toBeNull();
  expect(variables({ ...trace, operations: [] }).map((v) => v.nodeId)).toEqual([
    "input-t0",
  ]);
});

test("at a playback step, names hold that step's state; later ones wait", () => {
  const trace = {
    input_ids: ["t0"],
    tensors: {
      t0: tensor("t0", "x", "s0", "input"),
      t1: tensor("t1", "h", "s1"),
      t2: tensor("t2", "h", "s2"),
      t3: tensor("t3", "y", "s3"),
    },
    operations: [
      {
        id: "op0",
        index: 0,
        kind: "relu",
        outputs: ["t1"],
        source: { line: 1 },
      },
      {
        id: "op1",
        index: 1,
        kind: "tanh",
        outputs: ["t2"],
        source: { line: 2 },
      },
      {
        id: "op2",
        index: 2,
        kind: "sum",
        outputs: ["t3"],
        source: { line: 3 },
      },
    ],
  } as unknown as Run["trace"];
  const at = (through: number) =>
    variables(trace, through).map(
      (v) =>
        `${v.name}:${v.tensor.id}${v.pending ? ":pending" : ""}${v.fresh ? ":fresh" : ""}`,
    );
  expect(at(0)).toEqual(["x:t0", "h:t1:fresh", "y:t3:pending"]);
  expect(at(1)).toEqual(["x:t0", "h:t2:fresh", "y:t3:pending"]);
  expect(at(2)).toEqual(["x:t0", "h:t2", "y:t3:fresh"]);
  // Before the first step only the input holds a value.
  expect(at(-1)).toEqual(["x:t0", "h:t1:pending", "y:t3:pending"]);
  // Each name keeps its whole history; the position picks the state shown.
  const h = (through: number) => variables(trace, through)[1];
  expect(h(0).history.map((state) => state.tensor.id)).toEqual(["t1", "t2"]);
  expect([h(-1).shown, h(0).shown, h(1).shown, h(2).shown]).toEqual([
    -1, 0, 1, 1,
  ]);
});

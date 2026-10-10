import { expect, test } from "vitest";
import type { Run } from "../api/client";
import {
  filterShelf,
  keepUnchanged,
  pinFirst,
  sortShelf,
  variables,
  type Variable,
} from "./variables";

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

test("a step keeps the objects of the names it does not change", () => {
  const trace = {
    input_ids: ["t0"],
    tensors: {
      t0: tensor("t0", "x", "s0", "input"),
      t1: tensor("t1", "h", "s1"),
      t2: tensor("t2", "y", "s2"),
      t3: tensor("t3", "z", "s3"),
    },
    operations: ["t1", "t2", "t3"].map((output, index) => ({
      id: `op${index}`,
      index,
      kind: "relu",
      outputs: [output],
      source: { line: index + 1 },
    })),
  } as unknown as Run["trace"];
  const before = variables(trace, 0);
  const after = keepUnchanged(before, variables(trace, 1));
  // x stays as it was; h is no longer fresh, y is just written.
  expect(after.map((v, i) => v === before[i])).toEqual([
    true,
    false,
    false,
    true,
  ]);
  expect(after.map((v) => `${v.name}${v.fresh ? ":fresh" : ""}`)).toEqual([
    "x",
    "h",
    "y:fresh",
    "z",
  ]);
  // A state that reads the same but is another object, as in another run,
  // is shown as itself.
  const copied = { ...trace.tensors.t0 };
  const other = variables(trace, 1);
  other[0] = { ...other[0], tensor: copied };
  expect(keepUnchanged(after, other)[0].tensor).toBe(copied);
});

const item = (
  name: string,
  shape: number[],
  extra: Partial<Variable["tensor"]> = {},
  pending = false,
) =>
  ({
    name,
    pending,
    tensor: {
      name,
      shape,
      numel: shape.reduce((a, b) => a * b, 1),
      dtype: "float32",
      axes: shape.map((_, i) => (i ? "features" : "batch")),
      ...extra,
    },
  }) as unknown as Variable;

test("the shelf filters by name, dtype, axis name, or shape", () => {
  const shelf = [
    item("x", [2, 16]),
    item("mask", [2, 16], { dtype: "bool" }),
    item("logits", [2, 10]),
  ];
  const names = (query: string) =>
    filterShelf(shelf, query).map((variable) => variable.name);
  expect(names("bool")).toEqual(["mask"]);
  expect(names("2x16")).toEqual(["x", "mask"]);
  expect(names("[2, 10]")).toEqual(["logits"]);
  expect(names("LOG features")).toEqual(["logits"]);
  expect(names(" ")).toHaveLength(3);
  const faulty = [
    ...shelf,
    item("ratio", [2], {
      histogram: { non_finite: 1 } as Variable["tensor"]["histogram"],
    }),
  ];
  expect(filterShelf(faulty, "nan").map((variable) => variable.name)).toEqual([
    "ratio",
  ]);
});

test("the shelf sorts by size, value, or name, and pending names stay last", () => {
  const shelf = [
    item("b", [4], { minimum: -9, maximum: 1 }),
    item("later", [100], {}, true),
    item("a", [8], { minimum: 0, maximum: 3 }),
    item("c", [2], { minimum: null, maximum: null }),
  ];
  const order = (sort: Parameters<typeof sortShelf>[1]) =>
    sortShelf(shelf, sort).map((variable) => variable.name);
  expect(order("order")).toEqual(["b", "later", "a", "c"]);
  expect(order("size")).toEqual(["a", "b", "c", "later"]);
  expect(order("magnitude")).toEqual(["b", "a", "c", "later"]);
  expect(order("name")).toEqual(["a", "b", "c", "later"]);
  // By change: the largest share of changed values first, unknown last.
  const change = new Map([
    ["a", 0.25],
    ["b", 1],
    ["c", NaN],
  ]);
  expect(
    sortShelf(shelf, "change", change).map((variable) => variable.name),
  ).toEqual(["b", "a", "c", "later"]);
});

test("pinned names come first, unless they are not computed yet", () => {
  const shelf = [
    item("a", [1]),
    item("b", [1]),
    item("later", [1], {}, true),
    item("c", [1]),
  ];
  expect(
    pinFirst(shelf, new Set(["c", "later", "gone"])).map(
      (variable) => variable.name,
    ),
  ).toEqual(["c", "a", "b", "later"]);
  expect(pinFirst(shelf, new Set())).toBe(shelf);
});

test("a folded card played as one step marks every name it wrote fresh", () => {
  const trace = {
    input_ids: ["x"],
    tensors: {
      x: { id: "x", name: "x", role: "input" },
      a: { id: "a", name: "q" },
      b: { id: "b", name: "k" },
      c: { id: "c", name: "out" },
    },
    operations: [
      { id: "op0", index: 0, kind: "linear", outputs: ["a"], inputs: ["x"] },
      { id: "op1", index: 1, kind: "linear", outputs: ["b"], inputs: ["x"] },
      {
        id: "op2",
        index: 2,
        kind: "matmul",
        outputs: ["c"],
        inputs: ["a", "b"],
      },
    ],
  } as unknown as Parameters<typeof variables>[0];
  const fresh = (items: ReturnType<typeof variables>) =>
    items.filter((item) => item.fresh).map((item) => item.name);
  // One step: only its own result.
  expect(fresh(variables(trace, 1))).toEqual(["k"]);
  // A card of op0 and op1, played as one step, wrote both.
  expect(fresh(variables(trace, 1, -1))).toEqual(["q", "k"]);
  expect(
    variables(trace, 1, -1).find((item) => item.name === "out")?.pending,
  ).toBe(true);
});

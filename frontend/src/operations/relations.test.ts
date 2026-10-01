import { expect, test } from "vitest";
import type { Operation, Tensor } from "../api/client";
import { ravel, unravel } from "../tensors/coordinates";
import {
  broadcastIndex,
  reductionGroup,
  reductionTarget,
  relationSource,
  relationTargets,
  tensorRelation,
  triangleKeeps,
} from "./relations";

const tensor = (shape: number[], extra: Partial<Tensor> = {}) =>
  ({
    id: shape.join("x"),
    shape,
    numel: shape.reduce((a, b) => a * b, 1),
    dtype: "float32",
    ...extra,
  }) as Tensor;
const operation = (relation: object | null, extra: object = {}) =>
  ({
    kind: "op",
    status: "ok",
    arguments: {},
    lesson: { interaction: "relation", relation, mapping: null },
    ...extra,
  }) as unknown as Operation;
const slice = (start: number, step: number, out: number) => ({
  kind: "slice",
  start,
  step,
  out,
});

test("an index rule follows integers, slices, and inserted axes", () => {
  // x[1, None, 1:, ::2] on [2, 3, 4] → [1, 2, 2]
  const input = tensor([2, 3, 4]),
    output = tensor([1, 2, 2]);
  const op = operation({
    rule: "index",
    operand: 0,
    axes: [{ kind: "int", index: 1 }, slice(1, 1, 1), slice(0, 2, 2)],
  });
  const relation = tensorRelation(op, [input], output)!;
  const sources = [0, 1, 2, 3].map((t) =>
    unravel(relationSource(relation, input, output, t)!, input.shape),
  );
  expect(sources).toEqual([
    [1, 1, 0],
    [1, 1, 2],
    [1, 2, 0],
    [1, 2, 2],
  ]);
  const back = (coords: number[]) =>
    relationTargets(relation, input, output, ravel(coords, input.shape), 0);
  expect(back([1, 2, 2])).toEqual([3]);
  expect(back([0, 2, 2])).toEqual([]); // another batch
  expect(back([1, 0, 0])).toEqual([]); // before the slice start
  expect(back([1, 1, 1])).toEqual([]); // skipped by the step
});

test("index rules inconsistent with the shapes are rejected", () => {
  const input = tensor([2, 3, 4]);
  const make = (axes: object[], output: number[]) =>
    tensorRelation(
      operation({ rule: "index", operand: 0, axes }),
      [input],
      tensor(output),
    );
  const ok = [slice(0, 1, 0), slice(0, 1, 1), slice(0, 1, 2)];
  expect(make(ok, [2, 3, 4])).not.toBeNull();
  expect(make(ok.slice(0, 2), [2, 3])).toBeNull(); // an input axis is missing
  expect(make([ok[0], ok[1], slice(2, 1, 2)], [2, 3, 4])).toBeNull(); // overruns
  expect(make([ok[0], ok[1], slice(0, 1, 1)], [2, 3, 4])).toBeNull(); // reused axis
  expect(make([{ kind: "int", index: 2 }, ok[1], ok[2]], [3, 4])).toBeNull();
  expect(
    make(
      [{ kind: "int", index: 0 }, slice(0, 1, 1), slice(0, 1, 2)],
      [2, 3, 4],
    ),
  ).toBeNull(); // an unfilled output axis must have size one
  expect(
    tensorRelation(operation(null), [input], tensor([2, 3, 4])),
  ).toBeNull();
  expect(
    tensorRelation(
      operation({ rule: "tile", operand: 0 }, { status: "error" }),
      [input],
      input,
    ),
  ).toBeNull();
});

test("expand and repeat read input cells modulo their size", () => {
  const input = tensor([1, 3]),
    output = tensor([2, 2, 6]);
  const relation = tensorRelation(
    operation({ rule: "tile", operand: 0 }),
    [input],
    output,
  )!;
  expect(
    [0, 4, 11, 23].map((t) => relationSource(relation, input, output, t)),
  ).toEqual([0, 1, 2, 2]);
  // Selecting input cell 2 stays in the current output block.
  expect(relationTargets(relation, input, output, 2, 15)).toEqual([17]);
  expect(
    tensorRelation(
      operation({ rule: "tile", operand: 0 }),
      [tensor([2, 3])],
      tensor([2, 4]),
    ),
  ).toBeNull();
});

test("a table rule uses the recorded map for its own operand", () => {
  const ids = tensor([2]),
    table = tensor([3, 2]),
    output = tensor([2, 2]);
  const op = operation({ rule: "table", operand: 1 });
  op.lesson.mapping = [4, 5, 0, 1];
  const relation = tensorRelation(op, [ids, table], output)!;
  expect(relation.operand).toBe(1);
  expect(relationSource(relation, table, output, 2)).toBe(0);
  expect(relationTargets(relation, table, output, 5, 0)).toEqual([1]);
  op.lesson.mapping = [4, 5, 0, 6];
  expect(tensorRelation(op, [ids, table], output)).toBeNull();
  op.lesson.mapping = [4, 5, 0];
  expect(tensorRelation(op, [ids, table], output)).toBeNull();
});

test("reduction groups cover exactly the reduced axes", () => {
  const input = tensor([2, 3, 4]);
  const over = (axes: number[], keepdim: boolean, shape: number[]) => {
    const output = tensor(shape);
    const relation = tensorRelation(
      operation({ rule: "reduce", operand: 0, axes, keepdim }),
      [input],
      output,
    ) as Extract<ReturnType<typeof tensorRelation>, { rule: "reduce" }>;
    return { relation, output };
  };
  const last = over([2], false, [2, 3]);
  expect(reductionGroup(last.relation, input, last.output, 4)).toEqual({
    indices: [16, 17, 18, 19],
    size: 4,
  });
  expect(reductionTarget(last.relation, input, last.output, 18)).toBe(4);
  const kept = over([0, 2], true, [1, 3, 1]);
  const group = reductionGroup(kept.relation, input, kept.output, 1);
  expect(group.size).toBe(8);
  expect(group.indices.map((i) => unravel(i, input.shape)[1])).toEqual(
    Array(8).fill(1),
  );
  expect(reductionTarget(kept.relation, input, kept.output, 23)).toBe(2);
  const all = over([0, 1, 2], false, []);
  expect(reductionGroup(all.relation, input, all.output, 0, 5)).toEqual({
    indices: [0, 1, 2, 3, 4],
    size: 24,
  });
  expect(over([2], false, [2, 4]).relation).toBeNull();
  expect(over([2, 2], false, [2, 3]).relation).toBeNull();
});

test("elementwise rules align every operand by broadcasting", () => {
  const output = tensor([2, 3, 4]);
  expect(broadcastIndex([3, 1], output.shape, 23)).toBe(2);
  expect(broadcastIndex([4], output.shape, 22)).toBe(2);
  expect(broadcastIndex([], output.shape, 22)).toBe(0);
  const relation = tensorRelation(
    operation({ rule: "elementwise", operand: 0, roles: ["input", "other"] }),
    [output, tensor([3, 1])],
    output,
  )!;
  expect(relation.rule).toBe("elementwise");
  // A column cell maps to the current row position of the output.
  expect(relationTargets(relation, tensor([3, 1]), output, 2, 13)).toEqual([
    ravel([1, 2, 1], output.shape),
  ]);
  expect(
    tensorRelation(
      operation({ rule: "elementwise", operand: 0, roles: ["input", "other"] }),
      [output, tensor([3])],
      output,
    ),
  ).toBeNull();
  expect(
    tensorRelation(
      operation({ rule: "elementwise", operand: 0, roles: ["input"] }),
      [output, output],
      output,
    ),
  ).toBeNull();
  expect(triangleKeeps("tril", 0, [0, 1, 1])).toBe(true);
  expect(triangleKeeps("tril", 0, [0, 1, 2])).toBe(false);
  expect(triangleKeeps("triu", 1, [2, 2])).toBe(false);
  expect(triangleKeeps("tril", -1, [2, 1])).toBe(true);
});

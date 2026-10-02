import { expect, test } from "vitest";
import type { Operation, Tensor } from "../api/client";
import {
  broadcastConflicts,
  diagnose,
  diagnoseError,
  recordedNames,
} from "./diagnosis";

const tensors = (...shapes: number[][]) =>
  Object.fromEntries(
    shapes.map((shape, i) => [
      `t${i}`,
      {
        id: `t${i}`,
        name: "abcdef"[i],
        shape,
        numel: shape.reduce((x, y) => x * y, 1),
      } as Tensor,
    ]),
  );
const failed = (kind: string, count: number, args = {}, error = "failed") =>
  ({
    kind,
    status: "error",
    error,
    arguments: args,
    inputs: Array.from({ length: count }, (_, i) => `t${i}`),
  }) as unknown as Operation;

test("matrix products name the inner sizes and a transpose that fixes them", () => {
  const result = diagnose(failed("matmul", 2), tensors([2, 3, 4], [2, 3, 4]))!;
  expect(result.operands.map((o) => o.marks)).toEqual([[2], [1]]);
  expect(result.suggestion).toContain(
    "b.transpose(-2, -1) has shape [2, 4, 3]",
  );
  const batch = diagnose(failed("matmul", 2), tensors([2, 3, 4], [5, 4, 6]))!;
  expect(batch.title).toContain("batch");
  expect(batch.operands.map((o) => o.marks)).toEqual([[0], [0]]);
  const vector = diagnose(failed("matmul", 2), tensors([3, 4], [5]))!;
  expect(vector.operands[1].marks).toEqual([0]);
});

test("broadcast failures mark the conflicting axes and offer a unit axis", () => {
  expect(broadcastConflicts([2, 3, 4], [3, 1])).toEqual([]);
  const result = diagnose(failed("add", 2), tensors([2, 3, 4], [3]))!;
  expect(result.operands.map((o) => o.marks)).toEqual([[2], [0]]);
  expect(result.suggestion).toContain("b.unsqueeze(-1) has shape [3, 1]");
  const none = diagnose(failed("mul", 2), tensors([2, 3], [4, 5]))!;
  expect(none.suggestion).toBeNull();
  expect(none.operands[0].marks).toEqual([1, 0]);
});

test("reshape failures compare element counts", () => {
  const fixed = diagnose(
    failed("reshape", 1, { shape: [5, 5] }),
    tensors([2, 3, 4]),
  )!;
  expect(fixed.explanation).toContain("holds 24");
  expect(fixed.explanation).toContain("needs 25");
  const inferred = diagnose(
    failed("view", 1, { shape: [-1, 7] }),
    tensors([2, 3, 4]),
  )!;
  expect(inferred.explanation).toContain("divided evenly by 7");
});

test("joins, linear layers, convolutions, and axis errors are explained", () => {
  const cat = diagnose(
    failed("cat", 2, { dim: 0 }),
    tensors([2, 3, 4], [2, 2, 4]),
  )!;
  expect(cat.operands.map((o) => o.marks)).toEqual([[1], [1]]);
  const stack = diagnose(failed("stack", 2), tensors([2, 3, 4], [3, 4]))!;
  expect(stack.operands[1].marks).toEqual([0, 1]);
  const linear = diagnose(failed("linear", 2), tensors([2, 3, 4], [5, 3]))!;
  expect(linear.operands.map((o) => o.marks)).toEqual([[2], [1]]);
  const conv = diagnose(failed("conv2d", 2), tensors([2, 3, 4], [2, 3, 3, 3]))!;
  expect(conv.operands[0].marks).toEqual([0]);
  expect(conv.explanation).toContain("expects 3 input channels");
  expect(conv.suggestion).toBeNull();
  // Adding batch does not change the erroneous channel count.
  const batched = diagnose(
    failed("conv2d", 2),
    tensors([1, 2, 3, 4], [2, 3, 3, 3]),
  )!;
  expect(batched.title).toBe(conv.title);
  expect(batched.operands[0].marks).toEqual([1]);
  const permute = diagnose(
    failed("permute", 1, { dims: [0, 1] }),
    tensors([2, 3, 4]),
  )!;
  expect(permute.suggestion).toContain("permute(2, 1, 0)");
  const axis = diagnose(
    failed("transpose", 1, {}, "Dimension out of range (expected…)"),
    tensors([2, 3, 4]),
  )!;
  expect(axis.explanation).toContain("numbered 0 to 2");
});

test("unknown failures keep PyTorch's message and successes have no diagnosis", () => {
  const other = diagnose(
    failed("einsum", 2, {}, "bad subscript"),
    tensors([2], [2]),
  )!;
  expect(other.explanation).toBe("bad subscript");
  expect(other.operands).toHaveLength(2);
  expect(
    diagnose(
      { ...failed("add", 2), status: "ok" } as Operation,
      tensors([2], [3]),
    ),
  ).toBeNull();
  // A shape-compatible call that failed for another reason is not misdiagnosed.
  expect(
    diagnose(failed("add", 2, {}, "dtype"), tensors([2], [2]))!.title,
  ).toBe("This operation stopped the run");
});

const typedTensors = (
  ...specs: [shape: number[], dtype: string, values?: number[]][]
) =>
  Object.fromEntries(
    specs.map(([shape, dtype, values], i) => [
      `t${i}`,
      {
        id: `t${i}`,
        name: "abcdef"[i],
        shape,
        dtype,
        values: values ?? [],
        numel: shape.reduce((x, y) => x * y, 1),
      } as Tensor,
    ]),
  );

test("a view after a permute is explained by memory order, with reshape as the fix", () => {
  const result = diagnose(
    failed(
      "view",
      1,
      { shape: [2, 12] },
      "view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.",
    ),
    typedTensors([[2, 4, 3], "float32"]),
  )!;
  expect(result.title).toContain("reading order");
  expect(result.suggestion).toContain("a.reshape(2, 12)");
  expect(result.suggestion).toContain("a.contiguous().view(2, 12)");
  // A view with the wrong element count is still a count problem.
  expect(
    diagnose(
      failed("view", 1, { shape: [5, 5] }, "shape '[5, 5]' is invalid"),
      typedTensors([[2, 4, 3], "float32"]),
    )!.title,
  ).toBe("The element count does not fit");
});

test("data type failures name the operand to convert", () => {
  const index = diagnose(
    failed(
      "embedding",
      2,
      {},
      "Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead",
    ),
    typedTensors([[4], "float32"], [[5, 3], "float32"]),
  )!;
  expect(index.title).toBe("Positions must be whole numbers");
  expect(index.suggestion).toContain("a.long()");
  const target = diagnose(
    failed(
      "cross_entropy",
      2,
      {},
      "expected target dtype to be Long or Byte, but got Float",
    ),
    typedTensors([[2, 3], "float32"], [[2], "float32"]),
  )!;
  expect(target.suggestion).toContain("b.long()");
  const mixed = diagnose(
    failed("matmul", 2, {}, "expected scalar type Long but found Float"),
    typedTensors([[2, 3, 4], "int64"], [[2, 4, 3], "float32"]),
  )!;
  expect(mixed.title).toBe("The operands have different data types");
  expect(mixed.operands.map((o) => o.dtype)).toEqual(["int64", "float32"]);
  expect(mixed.suggestion).toBe("a.to(torch.float32) converts it to float32.");
  const widths = diagnose(
    failed(
      "linear",
      2,
      {},
      "expected m1 and m2 to have the same dtype, but got: double != float",
    ),
    typedTensors([[2, 3, 4], "float64"], [[5, 4], "float32"]),
  )!;
  expect(widths.suggestion).toBe("b.to(torch.float64) converts it to float64.");
  const mean = diagnose(
    failed(
      "mean",
      1,
      {},
      "mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Int",
    ),
    typedTensors([[2, 3], "int32"]),
  )!;
  expect(mean.suggestion).toContain("a.float().mean(");
  const condition = diagnose(
    failed(
      "where",
      3,
      {},
      "where expected condition to be a boolean tensor, but got a tensor with dtype Float",
    ),
    typedTensors([[2], "float32"], [[2], "float32"], [[2], "float32"]),
  )!;
  expect(condition.suggestion).toContain("a > 0");
});

test("out-of-range positions and impossible stretches are located", () => {
  const position = diagnose(
    failed(
      "__getitem__",
      1,
      {},
      "index 5 is out of bounds for dimension 1 with size 3",
    ),
    typedTensors([[2, 3, 4], "float32"]),
  )!;
  expect(position.operands[0].marks).toEqual([1]);
  expect(position.explanation).toContain("numbered 0 to 2");
  const table = diagnose(
    failed("embedding", 2, {}, "index out of range in self"),
    typedTensors([[2], "int64", [1, 7]], [[5, 3], "float32"]),
  )!;
  expect(table.explanation).toBe(
    "The table has 5 rows, numbered 0 to 4. The indices (a) include 7.",
  );
  const classes = diagnose(
    failed("cross_entropy", 2, {}, "Target 5 is out of bounds."),
    typedTensors([[2, 3], "float32"], [[2], "int64"]),
  )!;
  expect(classes.explanation).toContain("3 classes, numbered 0 to 2");
  const stretch = diagnose(
    failed(
      "expand",
      1,
      {},
      "The expanded size of the tensor (6) must match the existing size (3) at non-singleton dimension 1.  Target sizes: [2, 6, 4].  Tensor sizes: [2, 3, 4]",
    ),
    typedTensors([[2, 3, 4], "float32"]),
  )!;
  expect(stretch.operands[0].marks).toEqual([1]);
  expect(stretch.suggestion).toBe(
    "a.repeat(1, 2, 1) copies the values to reach [2, 6, 4].",
  );
  const uneven = diagnose(
    failed(
      "expand",
      1,
      {},
      "The expanded size of the tensor (5) must match the existing size (3) at non-singleton dimension 1.  Target sizes: [2, 5, 4].  Tensor sizes: [2, 3, 4]",
    ),
    typedTensors([[2, 3, 4], "float32"]),
  )!;
  expect(uneven.suggestion).toBeNull();
  const split = diagnose(
    failed(
      "unflatten",
      1,
      {},
      "unflatten: Provided sizes [2, 2] don't multiply up to the size of dim 1 (3) in the input tensor",
    ),
    typedTensors([[2, 3, 4], "float32"]),
  )!;
  expect(split.operands[0].marks).toEqual([1]);
});

test("a result without a variable is described, not used as code", () => {
  const all = tensors([2, 3], [3, 2]);
  all.t0.name = "long";
  all.t0.dtype = "int64";
  all.t1.dtype = "float32";
  const producers = [
    { kind: "long", outputs: ["t0"] },
    { kind: "transpose", outputs: ["t1"] },
  ] as unknown as Operation[];
  const typed = diagnose(
    failed("matmul", 2, {}, "expected scalar type Long but found Float"),
    { ...all, t1: { ...all.t1, name: "transpose" } },
    producers,
  )!;
  expect(typed.explanation).toContain("the long() result, is int64");
  expect(typed.suggestion).toBe(
    ".to(torch.float32) on the long() result converts it to float32.",
  );
});

test("axis errors say which axis was asked for", () => {
  const range = (kind: string, low: number, high: number, got: number) =>
    diagnose(
      failed(
        kind,
        1,
        {},
        `Dimension out of range (expected to be in range of [${low}, ${high}], but got ${got})`,
      ),
      tensors([2, 3, 4]),
    )!;
  expect(range("softmax", -3, 2, 3).explanation).toContain(
    "softmax was asked for axis 3, but a has 3 axes",
  );
  expect(range("softmax", -3, 2, 3).explanation).toContain(
    "the last axis is 2",
  );
  // unsqueeze may insert after the last axis.
  const insert = range("unsqueeze", -4, 3, 5);
  expect(insert.title).toBe("There is no such place for a new axis");
  expect(insert.explanation).toContain("dim runs from 0 to 3");
});

test("joins name the clashing sizes", () => {
  const cat = diagnose(
    failed("cat", 2, { dim: 2 }),
    tensors([2, 3, 4], [2, 4, 4]),
  )!;
  expect(cat.explanation).toContain("Axis 1 is 3 in a but 4 in tensor 1 (b).");
  const stack = diagnose(failed("stack", 2), tensors([2, 3, 4], [2, 2, 4]))!;
  expect(stack.explanation).toContain(
    "a is [2, 3, 4], but tensor 1 (b) is [2, 2, 4].",
  );
});

test("Python errors before any tensor step are explained too", () => {
  const name = diagnoseError(
    { type: "NameError", message: "name 'frist' is not defined", line: 2 },
    ["x", "first"],
  )!;
  expect(name.title).toBe("frist is not defined");
  expect(name.suggestion).toBe("Did you mean first?");
  const method = diagnoseError(
    {
      type: "AttributeError",
      message: "'Tensor' object has no attribute 'reshap'",
    },
    [],
  )!;
  expect(method.suggestion).toBe("Did you mean .reshape?");
  expect(diagnoseError({ type: "ValueError", message: "?" }, [])).toBeNull();
  const names = recordedNames({
    input_ids: ["t0"],
    operations: [
      { kind: "mul", outputs: ["t1"] },
      { kind: "add", outputs: ["t2"] },
    ] as unknown as Operation[],
    tensors: {
      t0: { name: "x" } as Tensor,
      t1: { name: "first" } as Tensor,
      t2: { name: "add" } as Tensor,
    },
  });
  expect(names).toEqual(["x", "first"]);
});

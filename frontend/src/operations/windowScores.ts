import type { ModuleCall, Operation, Run, Tensor } from "../api/client";
import { ravel, unravel } from "../tensors/coordinates";

export type WindowScores = {
  call: ModuleCall;
  operations: Operation[];
  scores: Tensor;
  bias: Tensor;
  table: Tensor;
  lookup: Tensor;
  mask: Tensor;
  weights: Tensor;
  windows: number;
  size: number;
};
const equal = (a: number[], b: number[]) =>
  a.length === b.length && a.every((v, i) => v === b[i]);
/** Qualify the recorded computation, never a module name in isolation. */
export function windowScores(run: Run, call: ModuleCall): WindowScores | null {
  if (
    call.module_type !== "WindowScoreAdjustment" ||
    call.end_index - call.start_index !== 7 ||
    call.inputs.length !== 1 ||
    call.outputs.length !== 1
  )
    return null;
  const ops = run.trace.operations.slice(call.start_index, call.end_index);
  const kinds = [
    "__getitem__",
    "permute",
    "add",
    "reshape",
    "masked_fill",
    "reshape",
    "softmax",
  ];
  if (
    ops.length !== 7 ||
    ops.some(
      (op, i) =>
        op.kind !== kinds[i] ||
        op.status !== "ok" ||
        op.mutations?.length ||
        op.outputs.length !== 1,
    )
  )
    return null;
  const get = (id: string) => run.trace.tensors[id];
  const scores = get(call.inputs[0]),
    table = get(ops[0].inputs[0]),
    lookup = get(ops[0].inputs[1]),
    bias = get(ops[1].outputs[0]),
    mask = get(ops[4].inputs[1]),
    weights = get(call.outputs[0]);
  if (
    !scores ||
    !table ||
    !lookup ||
    !bias ||
    !mask ||
    !weights ||
    !scores.numel ||
    scores.shape.length !== 4
  )
    return null;
  const [bw, heads, n, k] = scores.shape,
    size = Math.sqrt(n),
    windows = mask.shape[0];
  if (
    n !== k ||
    !Number.isInteger(size) ||
    !windows ||
    bw % windows ||
    !equal(weights.shape, scores.shape) ||
    !equal(bias.shape, [heads, n, n]) ||
    !equal(table.shape, [(2 * size - 1) ** 2, heads]) ||
    !equal(lookup.shape, [n, n]) ||
    !equal(mask.shape, [windows, 1, n, n]) ||
    !mask.dtype.endsWith("bool") ||
    !lookup.dtype.endsWith("int64")
  )
    return null;
  const inputs = [
    [table.id, lookup.id],
    [ops[0].outputs[0]],
    [scores.id, bias.id],
    [ops[2].outputs[0]],
    [ops[3].outputs[0], mask.id],
    [ops[4].outputs[0]],
    [ops[5].outputs[0]],
  ];
  if (
    ops.some((op, i) => op.inputs.join() !== inputs[i].join()) ||
    call.outputs[0] !== ops[6].outputs[0] ||
    String(ops[4].arguments.value) !== "-inf" ||
    Number(ops[6].arguments.dim) !== -1 ||
    Number(ops[2].arguments.alpha ?? 1) !== 1 ||
    !equal(ops[1].lesson.axis_order ?? [], [2, 0, 1])
  )
    return null;
  const shapes = [
    [n, n, heads],
    [heads, n, n],
    scores.shape,
    [bw / windows, windows, heads, n, n],
    [bw / windows, windows, heads, n, n],
    scores.shape,
    scores.shape,
  ];
  if (
    ops.some(
      (op, i) =>
        !get(op.outputs[0]) || !equal(get(op.outputs[0]).shape, shapes[i]),
    )
  )
    return null;
  if (
    [0, 1, 2, 3, 4, 5, 6].some(
      (i) => get(ops[i].outputs[0]).dtype !== scores.dtype,
    )
  )
    return null;
  if (
    table.dtype !== scores.dtype ||
    ![3, 5].every((i) => ops[i].lesson.mapping_rule === "identity")
  )
    return null;
  return {
    call,
    operations: ops,
    scores,
    bias,
    table,
    lookup,
    mask,
    weights,
    windows,
    size,
  };
}
export function findWindowScores(run: Run, operation: Operation) {
  if (!["masked_fill", "softmax"].includes(operation.kind)) return null;
  for (const call of run.trace.module_calls ?? []) {
    if (
      operation.index >= call.start_index &&
      operation.index < call.end_index
    ) {
      const lesson = windowScores(run, call);
      if (lesson) return lesson;
    }
  }
  return null;
}
export function windowScoreSelection(p: WindowScores, index: number) {
  const [bw, head, query, key] = unravel(index, p.weights.shape);
  return {
    batch: Math.floor(bw / p.windows),
    window: bw % p.windows,
    head,
    query,
    key,
    bias: ravel([head, query, key], p.bias.shape),
    mask: ravel([bw % p.windows, 0, query, key], p.mask.shape),
    lookup: ravel([query, key], p.lookup.shape),
    queryPosition: [Math.floor(query / p.size), query % p.size],
    keyPosition: [Math.floor(key / p.size), key % p.size],
  };
}
export function selectScoreOperand(
  p: WindowScores,
  index: number,
  from: "bias" | "mask",
  current: number,
) {
  const [bw, head] = unravel(current, p.weights.shape);
  if (from === "bias") {
    const [h, q, k] = unravel(index, p.bias.shape);
    return ravel([bw, h, q, k], p.weights.shape);
  }
  const [w, , q, k] = unravel(index, p.mask.shape);
  return ravel(
    [Math.floor(bw / p.windows) * p.windows + w, head, q, k],
    p.weights.shape,
  );
}

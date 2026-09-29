import type { Operation, Run, Tensor } from "../api/client";
import { ravel, unravel } from "../tensors/coordinates";

export type Addition = {
  left: Tensor;
  right: Tensor;
  output: Tensor;
  alpha: number;
};

/** Validate real operand shapes; never infer a residual or position embedding from size. */
export function tensorAddition(run: Run, op: Operation): Addition | null {
  if (
    op.kind !== "add" ||
    op.status !== "ok" ||
    op.mutations?.length ||
    op.inputs.length !== 2 ||
    op.outputs.length !== 1
  )
    return null;
  const [left, right] = op.inputs.map((id) => run.trace.tensors[id]);
  const output = run.trace.tensors[op.outputs[0]];
  const alpha = op.arguments?.alpha ?? 1;
  if (
    !left ||
    !right ||
    !output ||
    [left, right, output].some((t) => !t.numel) ||
    typeof alpha !== "number" ||
    !Number.isFinite(alpha)
  )
    return null;
  const rank = Math.max(left.shape.length, right.shape.length);
  if (output.shape.length !== rank) return null;
  for (let i = 1; i <= rank; i++) {
    const a = left.shape.at(-i) ?? 1,
      b = right.shape.at(-i) ?? 1;
    if (
      (a !== b && a !== 1 && b !== 1) ||
      output.shape.at(-i) !== Math.max(a, b)
    )
      return null;
  }
  return { left, right, output, alpha };
}

export function additionSelection(p: Addition, output: number) {
  const coordinates = unravel(output, p.output.shape);
  const operand = (tensor: Tensor) => {
    const offset = coordinates.length - tensor.shape.length;
    const coords = tensor.shape.map((size, axis) =>
      size === 1 ? 0 : coordinates[offset + axis],
    );
    return { coordinates: coords, index: ravel(coords, tensor.shape) };
  };
  return { coordinates, left: operand(p.left), right: operand(p.right) };
}

/** Preserve the current batch/slice on axes where many outputs reuse one input cell. */
export function additionInputSelection(
  p: Addition,
  side: "left" | "right",
  input: number,
  output: number,
) {
  const tensor = p[side];
  const coordinates = unravel(output, p.output.shape);
  const source = unravel(input, tensor.shape);
  const offset = coordinates.length - tensor.shape.length;
  tensor.shape.forEach((size, axis) => {
    if (size !== 1) coordinates[offset + axis] = source[axis];
  });
  return ravel(coordinates, p.output.shape);
}

export function broadcastAxes(tensor: Tensor, output: Tensor) {
  const offset = output.shape.length - tensor.shape.length;
  return output.shape.flatMap((size, axis) => {
    const source = tensor.shape[axis - offset] ?? 1;
    return size > 1 && source === 1 ? [axis] : [];
  });
}

export function finiteAddition(
  a: number | string | undefined,
  b: number | string | undefined,
  alpha: number,
) {
  if (
    typeof a !== "number" ||
    typeof b !== "number" ||
    ![a, b, alpha].every(Number.isFinite)
  )
    return undefined;
  const result = a + alpha * b;
  return Number.isFinite(result) ? result : undefined;
}

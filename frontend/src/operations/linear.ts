import type { Operation, Run, Tensor } from "../api/client";
import { ravel, unravel } from "../tensors/coordinates";

export type LinearProjection = {
  input: Tensor;
  weight: Tensor;
  bias?: Tensor;
  output: Tensor;
  features: number;
  outputs: number;
};

/** Validate recorded operands, including old runs predating the lesson capability. */
export function linearProjection(
  run: Run,
  op: Operation,
): LinearProjection | null {
  if (
    op.kind !== "linear" ||
    op.status !== "ok" ||
    op.outputs.length !== 1 ||
    ![2, 3].includes(op.inputs.length)
  )
    return null;
  const [input, weight, bias] = op.inputs.map((id) => run.trace.tensors[id]);
  const output = run.trace.tensors[op.outputs[0]];
  if (
    !input ||
    !weight ||
    !output ||
    (op.inputs.length === 3 && !bias) ||
    !input.shape.length ||
    weight.shape.length !== 2 ||
    [input, weight, output, ...(bias ? [bias] : [])].some((t) => !t.numel)
  )
    return null;
  const [outputs, features] = weight.shape;
  if (
    input.shape.at(-1) !== features ||
    output.shape.length !== input.shape.length ||
    output.shape.at(-1) !== outputs ||
    input.shape
      .slice(0, -1)
      .some((size, axis) => output.shape[axis] !== size) ||
    (bias && (bias.shape.length !== 1 || bias.shape[0] !== outputs))
  )
    return null;
  return { input, weight, bias, output, features, outputs };
}

export function linearSelection(
  p: LinearProjection,
  output: number,
  term: number,
) {
  const coordinates = unravel(output, p.output.shape);
  const prefix = coordinates.slice(0, -1);
  const feature = coordinates.at(-1)!;
  const inputCoordinates = [...prefix, term];
  const weightCoordinates = [feature, term];
  return {
    prefix,
    feature,
    output,
    coordinates,
    inputCoordinates,
    weightCoordinates,
    input: ravel(inputCoordinates, p.input.shape),
    weight: ravel(weightCoordinates, p.weight.shape),
  };
}

export function linearInputSelection(
  p: LinearProjection,
  input: number,
  output: number,
) {
  const coordinates = unravel(input, p.input.shape);
  return {
    output: ravel(
      [...coordinates.slice(0, -1), output % p.outputs],
      p.output.shape,
    ),
    term: coordinates.at(-1)!,
  };
}

export function linearWeightSelection(
  p: LinearProjection,
  weight: number,
  output: number,
) {
  const [feature, term] = unravel(weight, p.weight.shape);
  return { output: Math.floor(output / p.outputs) * p.outputs + feature, term };
}

/** Navigation allocates eight terms, even with billions of logical features. */
export function linearWindow(
  p: LinearProjection,
  output: number,
  term: number,
) {
  const start = Math.floor(term / 8) * 8;
  return Array.from({ length: Math.min(8, p.features - start) }, (_, i) => ({
    term: start + i,
    ...linearSelection(p, output, start + i),
  }));
}

export function finiteProduct(
  a: number | string | undefined,
  b: number | string | undefined,
) {
  if (typeof a !== "number" || typeof b !== "number") return undefined;
  const value = a * b;
  return Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(value)
    ? value
    : undefined;
}

/** Never treat absent or non-finite snapshot values as zero. */
export function contributionSum(products: (number | undefined)[]) {
  if (
    !products.length ||
    products.some((v) => v === undefined || !Number.isFinite(v))
  )
    return undefined;
  const sum = (products as number[]).reduce((total, value) => total + value, 0);
  return Number.isFinite(sum) ? sum : undefined;
}

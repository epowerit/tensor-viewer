export const product = (values: number[]) => values.reduce((a, b) => a * b, 1);

export function unravel(index: number, shape: number[]): number[] {
  const coords = Array(shape.length).fill(0);
  for (let i = shape.length - 1; i >= 0; i--) {
    coords[i] = index % shape[i];
    index = Math.floor(index / shape[i]);
  }
  return coords;
}

export function ravel(coords: number[], shape: number[]): number {
  return shape.reduce((offset, size, i) => offset * size + coords[i], 0);
}

export function formatValue(value: number | string | undefined): string {
  if (value === undefined) return "—";
  if (typeof value === "string") return value;
  if (Number.isInteger(value) && Math.abs(value) < 10000) return String(value);
  if (Math.abs(value) > 9999 || (value !== 0 && Math.abs(value) < 0.001))
    return value.toExponential(1);
  return Number(value.toFixed(3)).toString();
}

/** Operand batch coordinates for NumPy/PyTorch right-aligned broadcasting. */
export function broadcastBatch(
  outputBatch: number[],
  inputShape: number[],
): number[] {
  const shape = inputShape.slice(0, -2);
  const aligned = outputBatch.slice(outputBatch.length - shape.length);
  return shape.map((size, i) => (size === 1 ? 0 : aligned[i]));
}

export function dotContributors(
  leftShape: number[],
  rightShape: number[],
  outputShape: number[],
  index: number,
) {
  const coords = unravel(index, outputShape);
  const batch = coords.slice(0, -2);
  const row = coords.at(-2)!;
  const column = coords.at(-1)!;
  const k = leftShape.at(-1)!;
  return Array.from({ length: k }, (_, i) => ({
    left: ravel([...broadcastBatch(batch, leftShape), row, i], leftShape),
    right: ravel([...broadcastBatch(batch, rightShape), i, column], rightShape),
  }));
}

export function normalizationGroup(
  shape: number[],
  index: number,
  dimension: number,
): number[] {
  const axis = (dimension + shape.length) % shape.length;
  const coords = unravel(index, shape);
  return Array.from({ length: shape[axis] }, (_, i) =>
    ravel(
      coords.map((c, j) => (j === axis ? i : c)),
      shape,
    ),
  );
}

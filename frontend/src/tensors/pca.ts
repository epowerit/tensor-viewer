/** Rows projected onto their two main directions. */
export type Projection = {
  points: [number, number][];
  /** The share of the rows' spread along each direction. */
  explained: [number, number];
};

/** The top eigenvector of a symmetric matrix, by power iteration. */
function topDirection(
  matrix: Float64Array,
  size: number,
): [Float64Array, number] {
  let vector = new Float64Array(size).map((_, i) => 1 + i / size);
  let value = 0;
  for (let step = 0; step < 200; step++) {
    const next = new Float64Array(size);
    for (let row = 0; row < size; row++) {
      let sum = 0;
      for (let col = 0; col < size; col++)
        sum += matrix[row * size + col] * vector[col];
      next[row] = sum;
    }
    const norm = Math.hypot(...next);
    if (!norm) return [vector, 0];
    for (let i = 0; i < size; i++) next[i] /= norm;
    const moved = next.reduce((sum, v, i) => sum + Math.abs(v - vector[i]), 0);
    vector = next;
    value = norm;
    if (moved < 1e-10) break;
  }
  // One sign for a direction, so the same rows land the same way each time.
  let largest = 0;
  for (let i = 1; i < size; i++)
    if (Math.abs(vector[i]) > Math.abs(vector[largest])) largest = i;
  if (vector[largest] < 0) vector = vector.map((v) => -v);
  return [vector, value];
}

/**
 * Principal component analysis to two dimensions: the rows, centered, on
 * the two directions they spread along most, and how much of their spread
 * each direction holds.
 */
export function project2d(rows: ArrayLike<number>[]): Projection | null {
  const count = rows.length;
  const size = rows[0]?.length ?? 0;
  if (count < 2 || size < 2) return null;
  const mean = new Float64Array(size);
  for (const row of rows)
    for (let i = 0; i < size; i++) mean[i] += row[i] / count;
  const centered = rows.map((row) =>
    Float64Array.from({ length: size }, (_, i) => row[i] - mean[i]),
  );
  const covariance = new Float64Array(size * size);
  for (const row of centered)
    for (let a = 0; a < size; a++) {
      if (!row[a]) continue;
      for (let b = a; b < size; b++)
        covariance[a * size + b] += row[a] * row[b];
    }
  for (let a = 0; a < size; a++)
    for (let b = 0; b < a; b++)
      covariance[a * size + b] = covariance[b * size + a];
  let total = 0;
  for (let a = 0; a < size; a++) total += covariance[a * size + a];
  const [first, firstValue] = topDirection(covariance, size);
  // Deflate, then find the next direction.
  for (let a = 0; a < size; a++)
    for (let b = 0; b < size; b++)
      covariance[a * size + b] -= firstValue * first[a] * first[b];
  const [second, secondValue] = topDirection(covariance, size);
  const dot = (row: Float64Array, direction: Float64Array) =>
    row.reduce((sum, v, i) => sum + v * direction[i], 0);
  return {
    points: centered.map((row) => [dot(row, first), dot(row, second)]),
    explained: total
      ? [firstValue / total, Math.max(0, secondValue) / total]
      : [0, 0],
  };
}

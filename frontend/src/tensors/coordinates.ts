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

/**
 * A recorded value as its dtype holds it. A float32 value arrives as a
 * double (1.0410679578781128); it is shown with the fewest digits that read
 * back as the same float32 (1.041068). Booleans read True and False.
 */
export function exactValue(
  value: number | string | boolean | undefined,
  dtype: string,
): string {
  if (value === undefined) return "—";
  if (dtype === "bool") return Number(value) ? "True" : "False";
  if (typeof value !== "number" || !Number.isFinite(value))
    return String(value);
  if (Number.isInteger(value) && !dtype.startsWith("float"))
    return String(value);
  if (dtype === "float32") {
    const stored = Math.fround(value);
    for (let digits = 1; digits <= 9; digits++) {
      const text = Number(value.toPrecision(digits));
      if (Math.fround(text) === stored) return String(text);
    }
  }
  if (dtype === "float16") return String(Number(value.toPrecision(4)));
  if (dtype === "bfloat16") return String(Number(value.toPrecision(3)));
  return String(value);
}

/** Short, rounded cell labels; the inspector retains the full recorded value. */
export function formatCellValue(
  value: number | string | undefined,
  maxCharacters = 5,
): string {
  if (
    typeof value === "number" &&
    Number.isInteger(value) &&
    String(value).length <= maxCharacters
  )
    return String(value);
  const compact = (text: string) =>
    text
      .replace(/\.0+(?=e)/, "")
      .replace("e+", "e")
      .replace("Infinity", "∞");
  const initial = compact(formatValue(value));
  if (
    initial.length <= maxCharacters ||
    typeof value !== "number" ||
    !Number.isFinite(value)
  )
    return initial;
  for (const precision of [3, 2, 1]) {
    const label = compact(value.toPrecision(precision));
    if (label.length <= maxCharacters) return label;
  }
  return compact(value.toExponential(0));
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
  limit = Number.MAX_SAFE_INTEGER,
) {
  const coords = unravel(index, outputShape);
  const batch = coords.slice(0, -2);
  const row = coords.at(-2)!;
  const column = coords.at(-1)!;
  const k = leftShape.at(-1)!;
  return Array.from({ length: Math.min(k, limit) }, (_, i) => ({
    left: ravel([...broadcastBatch(batch, leftShape), row, i], leftShape),
    right: ravel([...broadcastBatch(batch, rightShape), i, column], rightShape),
  }));
}

export function normalizationGroup(
  shape: number[],
  index: number,
  dimension: number,
  limit = Number.MAX_SAFE_INTEGER,
): number[] {
  const axis = (dimension + shape.length) % shape.length;
  const coords = unravel(index, shape);
  return Array.from({ length: Math.min(shape[axis], limit) }, (_, i) =>
    ravel(
      coords.map((c, j) => (j === axis ? i : c)),
      shape,
    ),
  );
}

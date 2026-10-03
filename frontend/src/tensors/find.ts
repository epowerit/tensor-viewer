import type { Tensor } from "../api/client";

/** A recorded NaN or infinity: Python sends nan, inf, -inf as words. */
export function isBroken(value: unknown) {
  if (typeof value === "number") return !Number.isFinite(value);
  return (
    typeof value === "string" &&
    /^-?(nan|inf|infinity)$/.test(value.trim().toLowerCase())
  );
}

export type Landmarks = {
  /** Flat index of the first smallest and first largest finite value. */
  min: number | null;
  max: number | null;
  /** Flat indices of every NaN or infinite value, in order. */
  broken: number[];
};

/**
 * Where the smallest, largest and non-finite values sit, when every value
 * was recorded inline; null otherwise, since paged values arrive by window.
 */
export function landmarks(
  tensor: Pick<Tensor, "values" | "numel" | "dtype" | "value_source">,
): Landmarks | null {
  if (
    tensor.value_source === "shape" ||
    tensor.dtype === "bool" ||
    !tensor.numel ||
    tensor.values.length !== tensor.numel
  )
    return null;
  let min: number | null = null,
    max: number | null = null;
  const broken: number[] = [];
  tensor.values.forEach((raw, index) => {
    if (isBroken(raw)) {
      broken.push(index);
      return;
    }
    const value = Number(raw);
    if (!Number.isFinite(value)) return;
    if (min === null || value < Number(tensor.values[min])) min = index;
    if (max === null || value > Number(tensor.values[max])) max = index;
  });
  return { min, max, broken };
}

/** The next of `indices` after `from`, wrapping to the first. */
export function nextBroken(indices: number[], from: number) {
  return indices.find((index) => index > from) ?? indices[0] ?? null;
}

/** The previous of `indices` before `from`, wrapping to the last. */
export function previousOf(indices: number[], from: number) {
  return (
    [...indices].reverse().find((index) => index < from) ??
    indices.at(-1) ??
    null
  );
}

/**
 * The plane through `coords` (rows × columns, the other axes fixed) as its
 * own small tensor, so it can be copied as a Python list.
 */
export function planeValues(
  tensor: Pick<Tensor, "shape" | "values" | "numel" | "dtype">,
  plane: { row: number | null; column: number | null },
  coords: number[],
) {
  if (tensor.values.length !== tensor.numel) return null;
  const rows = plane.row === null ? 1 : tensor.shape[plane.row];
  const columns = plane.column === null ? 1 : tensor.shape[plane.column];
  const strides = tensor.shape.map((_, axis) =>
    tensor.shape.slice(axis + 1).reduce((a, b) => a * b, 1),
  );
  const values: (number | string | boolean)[] = [];
  for (let row = 0; row < rows; row++)
    for (let column = 0; column < columns; column++) {
      const at = coords.map((coordinate, axis) =>
        axis === plane.row ? row : axis === plane.column ? column : coordinate,
      );
      values.push(
        tensor.values[
          at.reduce(
            (flat, coordinate, axis) => flat + coordinate * strides[axis],
            0,
          )
        ],
      );
    }
  return {
    shape: plane.row === null ? [columns] : [rows, columns],
    dtype: tensor.dtype,
    values,
    numel: values.length,
  };
}

export type ValueQuery = {
  test: (value: number | string | boolean | undefined) => boolean;
  /** The query as understood, e.g. "|x| > 2". */
  label: string;
  /** The same query for the backend's search of a large tensor. */
  spec: Record<string, string | number | boolean>;
};

const TEST_NAMES: Record<string, string> = {
  ">": "gt",
  ">=": "ge",
  "<": "lt",
  "<=": "le",
  "==": "eq",
  "!=": "ne",
};

const COMPARE: Record<string, (a: number, b: number) => boolean> = {
  ">": (a, b) => a > b,
  ">=": (a, b) => a >= b,
  "<": (a, b) => a < b,
  "<=": (a, b) => a <= b,
  "==": (a, b) => a === b,
  "!=": (a, b) => a !== b,
};

/**
 * A value search: `> 0.5`, `<= -1`, `== 0`, `!= 0`, a bare number (equal to
 * it), `abs > 2` or `|x| > 2` for magnitudes, and `nan`, `inf` or `finite`.
 * Returns an error message when the text is not one of these.
 */
export function parseQuery(text: string): ValueQuery | { error: string } {
  const raw = text.trim().toLowerCase();
  if (!raw) return { error: "" };
  if (raw === "nan")
    return {
      label: "NaN",
      spec: { test: "nan" },
      test: (value) =>
        isBroken(value) && /nan/i.test(String(value).toLowerCase()),
    };
  if (raw === "inf" || raw === "infinity" || raw === "±inf")
    return {
      label: "±inf",
      spec: { test: "inf" },
      test: (value) => isBroken(value) && /inf/i.test(String(value)),
    };
  if (raw === "finite")
    return {
      label: "finite",
      spec: { test: "finite" },
      test: (value) => value !== undefined && !isBroken(value),
    };
  const match = raw.match(
    /^(abs|\|x\|)?\s*(>=|<=|==|!=|=|>|<)?\s*(-?(?:\d+\.?\d*|\.\d+)(?:e-?\d+)?)$/,
  );
  if (!match)
    return {
      error: "Try > 0.5, == 0, abs > 2, nan, or inf.",
    };
  const [, abs, written = "==", number] = match;
  const operator = written === "=" ? "==" : written;
  const target = Number(number);
  const compare = COMPARE[operator];
  return {
    label: `${abs ? "|x|" : "x"} ${operator} ${target}`,
    spec: {
      test: TEST_NAMES[operator],
      value: target,
      magnitude: !!abs,
    },
    test: (value) => {
      if (value === undefined || isBroken(value)) return false;
      const x = Number(value);
      return Number.isFinite(x) && compare(abs ? Math.abs(x) : x, target);
    },
  };
}

/** Flat indices of every recorded value the query matches, in order. */
export function findValues(
  tensor: Pick<Tensor, "values" | "numel">,
  query: ValueQuery,
) {
  if (tensor.values.length !== tensor.numel) return null;
  const found: number[] = [];
  tensor.values.forEach((value, index) => {
    if (query.test(value)) found.push(index);
  });
  return found;
}

/**
 * The 1st, 50th and 99th percentiles of the finite values, interpolated
 * linearly as numpy.percentile does; null without finite values.
 */
export function quantiles(values: (number | string | boolean)[]) {
  const finite = values
    .filter((value) => !isBroken(value))
    .map(Number)
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  if (!finite.length) return null;
  return [0.01, 0.5, 0.99].map((p) => {
    const position = p * (finite.length - 1);
    const below = Math.floor(position);
    const above = Math.min(finite.length - 1, below + 1);
    return finite[below] + (finite[above] - finite[below]) * (position - below);
  });
}

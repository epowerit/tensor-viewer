import { product, ravel, unravel } from "./coordinates";

export type Plane = { row: number | null; column: number | null };
export const PAGE_SIZE = 8;

/** Validate a whole logical coordinate without silently clipping any axis. */
export function parseCoordinate(
  text: string,
  shape: number[],
): { index: number | null; error: string } {
  if (!product(shape))
    return { index: null, error: "This tensor has no elements." };
  const raw = text
    .trim()
    .replace(/^\[(.*)\]$/, "$1")
    .trim();
  const parts = raw ? raw.split(/[\s,]+/) : [];
  if (parts.length !== shape.length)
    return {
      index: null,
      error: `Enter ${shape.length} ${shape.length === 1 ? "index" : "indices"}, one per axis.`,
    };
  // Negative indices count from the end, as in Python: -1 is the last.
  const coords = parts.map((part, axis) => {
    const value = Number(part);
    return value < 0 ? shape[axis] + value : value;
  });
  const invalid = coords.findIndex(
    (value, axis) =>
      !/^-?\d+$/.test(parts[axis]) ||
      !Number.isSafeInteger(value) ||
      value < 0 ||
      value >= shape[axis],
  );
  if (invalid >= 0)
    return {
      index: null,
      error: `Axis ${invalid} needs an integer from ${(-shape[invalid]).toLocaleString()} to ${(shape[invalid] - 1).toLocaleString()}.`,
    };
  const index = ravel(coords, shape);
  return Number.isSafeInteger(index)
    ? { index, error: "" }
    : {
        index: null,
        error: "This coordinate exceeds the supported index range.",
      };
}

export function defaultPlane(rank: number): Plane {
  return { row: rank > 1 ? rank - 2 : null, column: rank ? rank - 1 : null };
}

/** Selecting the other visible axis swaps the view, never the tensor data. */
export function changePlane(
  plane: Plane,
  side: keyof Plane,
  axis: number,
): Plane {
  const other = side === "row" ? "column" : "row";
  return {
    ...plane,
    [side]: axis,
    [other]: plane[other] === axis ? plane[side] : plane[other],
  };
}

export function hiddenAxes(shape: number[], plane: Plane) {
  return shape
    .map((_, axis) => axis)
    .filter((axis) => axis !== plane.row && axis !== plane.column);
}

export function planeSize(shape: number[], plane: Plane) {
  return {
    rows: plane.row === null ? 1 : shape[plane.row],
    columns: plane.column === null ? 1 : shape[plane.column],
  };
}

export function planeCoordinates(
  shape: number[],
  plane: Plane,
  fixed: number[],
  row: number,
  column: number,
) {
  return shape.map((_, axis) =>
    axis === plane.row
      ? row
      : axis === plane.column
        ? column
        : (fixed[axis] ?? 0),
  );
}

export function safeIndex(index: number, shape: number[]) {
  return Math.max(
    0,
    Math.min(Math.max(0, product(shape) - 1), Math.trunc(index)),
  );
}

export function coordinatesFor(index: number, shape: number[]) {
  return product(shape) === 0
    ? shape.map(() => 0)
    : unravel(safeIndex(index, shape), shape);
}

/**
 * Move in the displayed plane, respecting its real axis sizes and slices.
 * PageUp and PageDown step to the previous or next slice; PlaneStart and
 * PlaneEnd (Ctrl/⌘ + Home or End) go to the plane's first or last cell.
 */
export function moveInPlane(
  index: number,
  shape: number[],
  plane: Plane,
  key: string,
): number {
  if (!product(shape)) return 0;
  const coords = coordinatesFor(index, shape);
  if (key === "PageUp" || key === "PageDown") {
    // Page through slices along the innermost hidden axis, e.g. channels.
    const slice = hiddenAxes(shape, plane)
      .filter((axis) => shape[axis] > 1)
      .at(-1);
    if (slice === undefined) return index;
    coords[slice] = Math.max(
      0,
      Math.min(shape[slice] - 1, coords[slice] + (key === "PageUp" ? -1 : 1)),
    );
    return ravel(coords, shape);
  }
  if (key === "PlaneStart" || key === "PlaneEnd") {
    // The first or last cell of the plane on screen; slices stay put.
    for (const axis of [plane.row, plane.column])
      if (axis !== null)
        coords[axis] = key === "PlaneStart" ? 0 : shape[axis] - 1;
    return ravel(coords, shape);
  }
  const axis =
    key === "ArrowUp" || key === "ArrowDown" ? plane.row : plane.column;
  if (axis === null) return index;
  if (key === "Home") coords[axis] = 0;
  else if (key === "End") coords[axis] = shape[axis] - 1;
  else
    coords[axis] = Math.max(
      0,
      Math.min(
        shape[axis] - 1,
        coords[axis] + (key === "ArrowUp" || key === "ArrowLeft" ? -1 : 1),
      ),
    );
  return ravel(coords, shape);
}

/** Storage positions are element offsets; logical flat indices ignore strides. */
export function storagePosition(
  coords: number[],
  strides: number[],
  offset: number,
) {
  return coords.reduce(
    (position, coordinate, axis) => position + coordinate * strides[axis],
    offset,
  );
}

export function mappedOutputs(
  mapping: number[] | null | undefined,
  inputIndex: number,
): number[] {
  return (
    mapping?.flatMap((source, output) =>
      source === inputIndex ? [output] : [],
    ) ?? []
  );
}

export function axisColor(name: string, index: number) {
  if (/batch/.test(name)) return 0;
  if (/features|channels/.test(name)) return 3;
  if (/heads|groups/.test(name)) return 2;
  if (/tokens/.test(name)) return 1;
  return index % 4;
}

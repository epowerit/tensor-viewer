import type { Operation, Tensor } from "../api/client";
import { ravel, unravel } from "./coordinates";
import { mappedOutputs } from "./plane";

function rollIndex(
  op: Operation,
  shape: number[],
  index: number,
  direction: number,
) {
  const shifts = Array.isArray(op.arguments.shifts)
    ? op.arguments.shifts.map(Number)
    : [Number(op.arguments.shifts)];
  const raw = op.arguments.dims;
  const dims =
    raw == null ? [] : Array.isArray(raw) ? raw.map(Number) : [Number(raw)];
  const mod = (n: number, size: number) => ((n % size) + size) % size;
  if (!dims.length)
    return mod(
      index + direction * shifts.reduce((a, b) => a + b, 0),
      shape.reduce((a, b) => a * b, 1),
    );
  const coordinates = unravel(index, shape);
  dims.forEach((dim, i) => {
    const axis = mod(dim, shape.length);
    coordinates[axis] = mod(
      coordinates[axis] + direction * shifts[i],
      shape[axis],
    );
  });
  return ravel(coordinates, shape);
}

export function sourceIndex(
  op: Operation,
  input: Tensor,
  output: Tensor,
  index: number,
): number | undefined {
  if (op.lesson.mapping) return op.lesson.mapping[index];
  const rule = op.lesson.mapping_rule;
  if (rule === "identity") return index;
  if (rule === "roll") return rollIndex(op, input.shape, index, -1);
  const coords = unravel(index, output.shape);
  if (rule === "permutation" && op.lesson.axis_order) {
    const source = Array(input.shape.length).fill(0);
    op.lesson.axis_order.forEach((axis, i) => (source[axis] = coords[i]));
    return ravel(source, input.shape);
  }
  if (rule === "unfold") {
    const axis =
      (Number(op.arguments.dimension) + input.shape.length) %
      input.shape.length;
    const source = coords.slice(0, -1);
    source[axis] = source[axis] * Number(op.arguments.step) + coords.at(-1)!;
    return ravel(source, input.shape);
  }
}

/** Inverse relationships stay bounded; overlapping windows show up to 256 matches. */
export function outputIndices(
  op: Operation,
  input: Tensor,
  output: Tensor,
  index: number,
): number[] {
  if (op.lesson.mapping) return mappedOutputs(op.lesson.mapping, index);
  const rule = op.lesson.mapping_rule;
  if (rule === "identity") return [index];
  if (rule === "roll") return [rollIndex(op, input.shape, index, 1)];
  const coords = unravel(index, input.shape);
  if (rule === "permutation" && op.lesson.axis_order)
    return [
      ravel(
        op.lesson.axis_order.map((axis) => coords[axis]),
        output.shape,
      ),
    ];
  if (rule === "unfold") {
    const axis =
      (Number(op.arguments.dimension) + input.shape.length) %
      input.shape.length;
    const size = Number(op.arguments.size),
      step = Number(op.arguments.step);
    const first = Math.max(0, Math.ceil((coords[axis] - size + 1) / step));
    const last = Math.min(
      output.shape[axis] - 1,
      Math.floor(coords[axis] / step),
    );
    return Array.from(
      { length: Math.max(0, Math.min(256, last - first + 1)) },
      (_, i) => {
        const window = first + i;
        const target = [...coords, coords[axis] - window * step];
        target[axis] = window;
        return ravel(target, output.shape);
      },
    );
  }
  return [];
}

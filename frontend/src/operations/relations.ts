import type { Operation, Tensor } from "../api/client";
import { product, ravel, unravel } from "../tensors/coordinates";

type IndexAxis =
  | { kind: "int"; index: number }
  | { kind: "slice"; start: number; step: number; out: number };
export type Relation =
  | { rule: "index"; operand: number; axes: IndexAxis[] }
  | { rule: "tile"; operand: number }
  | { rule: "table"; operand: number; mapping: number[] }
  | { rule: "reduce"; operand: number; axes: number[]; keepdim: boolean }
  | { rule: "prefix"; operand: number; axis: number }
  | { rule: "pad"; operand: number; before: number[] }
  | { rule: "one_hot"; operand: number }
  | {
      rule: "channel_affine";
      operand: number;
      /** The axis whose position selects a statistic; other axes share it. */
      axis: number;
      /** Operand roles in order: input, running_mean, running_var, weight, bias. */
      roles: string[];
      eps: number;
    }
  | {
      rule: "class_loss";
      operand: number;
      reduction: "mean" | "sum" | "none";
      ignore_index: number;
      /** True when the scores are already log-probabilities (nll_loss). */
      normalized: boolean;
    }
  | {
      rule: "einsum";
      operand: number;
      /** One string of index letters per operand, and one for the output. */
      inputs: string[];
      output: string;
      sizes: Record<string, number>;
    }
  | {
      rule: "elementwise";
      operand: number;
      roles: string[];
      diagonal?: number;
      reversed?: boolean;
    };

const whole = (value: unknown): value is number =>
  typeof value === "number" && Number.isInteger(value);

/** Broadcasting aligns from the last axis; an axis of size one is reused. */
export function broadcastIndex(
  shape: number[],
  outputShape: number[],
  target: number,
): number {
  const coords = unravel(target, outputShape);
  const offset = outputShape.length - shape.length;
  return ravel(
    shape.map((size, axis) => (size === 1 ? 0 : coords[offset + axis])),
    shape,
  );
}
const broadcasts = (shape: number[], target: number[]) =>
  shape.length <= target.length &&
  shape.every((size, i) => {
    const other = target[target.length - shape.length + i];
    return size === other || size === 1;
  });

/** Accept a recorded rule only when it is consistent with the recorded shapes. */
export function tensorRelation(
  op: Operation,
  inputs: (Tensor | undefined)[],
  output: Tensor | undefined,
  /**
   * Accept the cell rule even when a dedicated lesson (activation, reduction)
   * presents the step; views keep requiring the rule's own lesson.
   */
  options: { anyLesson?: boolean } = {},
): Relation | null {
  const raw = op.lesson?.relation as Record<string, unknown> | null | undefined;
  if (
    !raw ||
    !output ||
    op.status !== "ok" ||
    (!options.anyLesson && op.lesson.interaction !== "relation") ||
    !output.numel ||
    product(output.shape) !== output.numel ||
    !whole(raw.operand)
  )
    return null;
  const operand = raw.operand;
  const input = inputs[operand];
  if (!input || !input.numel || product(input.shape) !== input.numel)
    return null;
  if (raw.rule === "index") {
    const axes = raw.axes as IndexAxis[];
    if (!Array.isArray(axes) || axes.length !== input.shape.length) return null;
    const used = new Set<number>();
    for (const [axis, item] of axes.entries()) {
      const size = input.shape[axis];
      if (item?.kind === "int") {
        if (!whole(item.index) || item.index < 0 || item.index >= size)
          return null;
      } else if (item?.kind === "slice") {
        const { start, step, out } = item;
        if (
          !whole(start) ||
          !whole(step) ||
          !whole(out) ||
          step < 1 ||
          start < 0 ||
          out < 0 ||
          out >= output.shape.length ||
          used.has(out) ||
          start + step * (output.shape[out] - 1) >= size
        )
          return null;
        used.add(out);
      } else return null;
    }
    // Axes that no slice fills were inserted with None and have one position.
    if (output.shape.some((size, axis) => !used.has(axis) && size !== 1))
      return null;
    return { rule: "index", operand, axes };
  }
  if (raw.rule === "tile") {
    const offset = output.shape.length - input.shape.length;
    if (
      offset < 0 ||
      input.dtype !== output.dtype ||
      input.shape.some((size, axis) => output.shape[offset + axis] % size)
    )
      return null;
    return { rule: "tile", operand };
  }
  if (raw.rule === "table") {
    const mapping = op.lesson.mapping;
    if (
      !mapping ||
      mapping.length !== output.numel ||
      mapping.some((i) => !whole(i) || i < 0 || i >= input.numel)
    )
      return null;
    return { rule: "table", operand, mapping };
  }
  if (raw.rule === "reduce") {
    const axes = raw.axes as number[];
    const rank = input.shape.length;
    if (
      !Array.isArray(axes) ||
      typeof raw.keepdim !== "boolean" ||
      new Set(axes).size !== axes.length ||
      axes.some((axis) => !whole(axis) || axis < 0 || axis >= rank)
    )
      return null;
    const expected = input.shape.flatMap((size, axis) =>
      axes.includes(axis) ? (raw.keepdim ? [1] : []) : [size],
    );
    if (expected.join() !== output.shape.join()) return null;
    return { rule: "reduce", operand, axes, keepdim: raw.keepdim };
  }
  if (raw.rule === "prefix") {
    if (
      !whole(raw.axis) ||
      raw.axis < 0 ||
      raw.axis >= input.shape.length ||
      input.shape.join() !== output.shape.join()
    )
      return null;
    return { rule: "prefix", operand, axis: raw.axis };
  }
  if (raw.rule === "pad") {
    const before = raw.before as number[];
    if (
      !Array.isArray(before) ||
      before.length !== input.shape.length ||
      output.shape.length !== input.shape.length ||
      before.some(
        (pad, axis) =>
          !whole(pad) ||
          pad < 0 ||
          output.shape[axis] < pad + input.shape[axis],
      )
    )
      return null;
    return { rule: "pad", operand, before };
  }
  if (raw.rule === "channel_affine") {
    const roles = raw.roles as string[];
    const axis = raw.axis;
    if (
      !Array.isArray(roles) ||
      roles.length !== inputs.length ||
      roles[0] !== "input" ||
      roles[1] !== "running_mean" ||
      roles[2] !== "running_var" ||
      !whole(axis) ||
      axis < 0 ||
      axis >= input.shape.length ||
      typeof raw.eps !== "number" ||
      !Number.isFinite(raw.eps) ||
      input.shape.join() !== output.shape.join() ||
      inputs
        .slice(1)
        .some(
          (tensor) =>
            !tensor ||
            tensor.shape.length !== 1 ||
            tensor.shape[0] !== input.shape[axis],
        )
    )
      return null;
    return { rule: "channel_affine", operand, axis, roles, eps: raw.eps };
  }
  if (raw.rule === "one_hot") {
    if (
      input.dtype !== "int64" ||
      output.shape.length !== input.shape.length + 1 ||
      output.shape.slice(0, -1).join() !== input.shape.join()
    )
      return null;
    return { rule: "one_hot", operand };
  }
  if (raw.rule === "class_loss") {
    const target = inputs[1];
    const reduction = raw.reduction;
    if (
      inputs.length !== 2 ||
      !target ||
      target.dtype !== "int64" ||
      (input.shape.length !== 1 && input.shape.length !== 2) ||
      target.shape.join() !== input.shape.slice(0, -1).join() ||
      (reduction !== "mean" && reduction !== "sum" && reduction !== "none") ||
      !whole(raw.ignore_index) ||
      output.shape.join() !== (reduction === "none" ? target.shape.join() : "")
    )
      return null;
    return {
      rule: "class_loss",
      operand,
      reduction,
      ignore_index: raw.ignore_index,
      normalized: raw.normalized === true,
    };
  }
  if (raw.rule === "einsum") {
    const terms = raw.inputs as string[];
    const sizes = raw.sizes as Record<string, number>;
    const result = raw.output;
    if (
      !Array.isArray(terms) ||
      terms.length !== inputs.length ||
      typeof result !== "string" ||
      !sizes ||
      typeof sizes !== "object" ||
      new Set(result).size !== result.length
    )
      return null;
    const fits = (letters: unknown, shape: number[]) =>
      typeof letters === "string" &&
      /^[A-Za-z]*$/.test(letters) &&
      letters.length === shape.length &&
      [...letters].every((letter, axis) => sizes[letter] === shape[axis]);
    if (
      !fits(result, output.shape) ||
      inputs.some((tensor, i) => !tensor || !fits(terms[i], tensor.shape))
    )
      return null;
    return { rule: "einsum", operand, inputs: terms, output: result, sizes };
  }
  if (raw.rule === "elementwise") {
    const roles = raw.roles as string[];
    if (
      !Array.isArray(roles) ||
      roles.length !== inputs.length ||
      inputs.some(
        (tensor) => !tensor || !broadcasts(tensor.shape, output.shape),
      )
    )
      return null;
    if (
      raw.diagonal !== undefined &&
      (!whole(raw.diagonal) || output.shape.length < 2)
    )
      return null;
    return {
      rule: "elementwise",
      operand,
      roles,
      diagonal: raw.diagonal as number | undefined,
      reversed: raw.reversed === true,
    };
  }
  return null;
}

/** The single input cell an output cell copies, for selection-like rules. */
export function relationSource(
  relation: Relation,
  input: Tensor,
  output: Tensor,
  target: number,
): number | undefined {
  if (relation.rule === "table") return relation.mapping[target];
  const out = unravel(target, output.shape);
  if (relation.rule === "index")
    return ravel(
      relation.axes.map((item) =>
        item.kind === "int"
          ? item.index
          : item.start + item.step * out[item.out],
      ),
      input.shape,
    );
  if (relation.rule === "tile") {
    const offset = output.shape.length - input.shape.length;
    return ravel(
      input.shape.map((size, axis) => out[offset + axis] % size),
      input.shape,
    );
  }
  if (relation.rule === "pad") {
    const coords = out.map((c, axis) => c - relation.before[axis]);
    // A border cell has no input cell behind it.
    return coords.every((c, axis) => c >= 0 && c < input.shape[axis])
      ? ravel(coords, input.shape)
      : undefined;
  }
}

/** Input cells accumulated into a running total, up to `limit`. */
export function prefixGroup(
  relation: Extract<Relation, { rule: "prefix" }>,
  input: Tensor,
  target: number,
  limit = 256,
): { indices: number[]; size: number } {
  const coords = unravel(target, input.shape);
  const size = coords[relation.axis] + 1;
  // Keep the most recent terms when the run is longer than the limit.
  const first = Math.max(0, size - limit);
  return {
    size,
    indices: Array.from({ length: size - first }, (_, i) =>
      ravel(
        coords.map((c, axis) => (axis === relation.axis ? first + i : c)),
        input.shape,
      ),
    ),
  };
}

/**
 * The products summed into one einsum output cell. Letters kept by the output
 * are fixed by the cell; every combination of the other letters is one term,
 * holding one cell index per operand.
 */
export function einsumTerms(
  relation: Extract<Relation, { rule: "einsum" }>,
  shapes: number[][],
  outputShape: number[],
  target: number,
  limit = 256,
): { terms: number[][]; size: number } {
  const fixed: Record<string, number> = {};
  unravel(target, outputShape).forEach(
    (c, axis) => (fixed[relation.output[axis]] = c),
  );
  const free = [...new Set(relation.inputs.join(""))].filter(
    (letter) => !(letter in fixed),
  );
  const ranges = free.map((letter) => relation.sizes[letter]);
  const size = product(ranges);
  const terms = Array.from({ length: Math.min(size, limit) }, (_, i) => {
    const values = { ...fixed };
    unravel(i, ranges).forEach((c, j) => (values[free[j]] = c));
    return relation.inputs.map((letters, operand) =>
      ravel(
        [...letters].map((letter) => values[letter]),
        shapes[operand],
      ),
    );
  });
  return { terms, size };
}

/** The output cell a reduced input cell contributes to. */
export function reductionTarget(
  relation: Extract<Relation, { rule: "reduce" }>,
  input: Tensor,
  output: Tensor,
  source: number,
): number {
  const coords = unravel(source, input.shape);
  return ravel(
    coords.flatMap((c, axis) =>
      relation.axes.includes(axis) ? (relation.keepdim ? [0] : []) : [c],
    ),
    output.shape,
  );
}

/** Input cells combined into one reduced output, bounded by `limit`. */
export function reductionGroup(
  relation: Extract<Relation, { rule: "reduce" }>,
  input: Tensor,
  output: Tensor,
  target: number,
  limit = 256,
): { indices: number[]; size: number } {
  const out = unravel(target, output.shape);
  let next = 0;
  const base = input.shape.map((_, axis) =>
    relation.axes.includes(axis)
      ? relation.keepdim
        ? (next++, 0)
        : 0
      : out[next++],
  );
  const sizes = relation.axes.map((axis) => input.shape[axis]);
  const size = product(sizes);
  const indices = Array.from({ length: Math.min(size, limit) }, (_, i) => {
    const free = unravel(i, sizes);
    const coords = [...base];
    relation.axes.forEach((axis, j) => (coords[axis] = free[j]));
    return ravel(coords, input.shape);
  });
  return { indices, size };
}

/** Output cells that read a selected input cell (at most `limit`). */
export function relationTargets(
  relation: Relation,
  input: Tensor,
  output: Tensor,
  source: number,
  current: number,
  limit = 256,
): number[] {
  if (relation.rule === "reduce")
    return [reductionTarget(relation, input, output, source)];
  if (relation.rule === "table")
    return relation.mapping
      .flatMap((from, target) => (from === source ? [target] : []))
      .slice(0, limit);
  const coords = unravel(source, input.shape);
  if (relation.rule === "one_hot") {
    // Stay on the class currently shown.
    const classes = output.shape.at(-1)!;
    return [source * classes + (current % classes)];
  }
  if (relation.rule === "class_loss")
    return [
      relation.reduction === "none" && input.shape.length === 2 ? coords[0] : 0,
    ];
  if (relation.rule === "prefix" || relation.rule === "channel_affine")
    return [source];
  if (relation.rule === "pad")
    return [
      ravel(
        coords.map((c, axis) => c + relation.before[axis]),
        output.shape,
      ),
    ];
  if (relation.rule === "einsum") {
    // Move along the output letters this operand carries; keep the rest.
    const out = unravel(current, output.shape);
    const fixed = new Map<string, number>();
    for (const [axis, letter] of [
      ...relation.inputs[relation.operand],
    ].entries()) {
      // Repeated letters select a diagonal. Off-diagonal cells do not
      // participate, even when that letter is reduced out of the result.
      if (fixed.has(letter) && fixed.get(letter) !== coords[axis]) return [];
      fixed.set(letter, coords[axis]);
      const at = relation.output.indexOf(letter);
      if (at >= 0) out[at] = coords[axis];
    }
    return [ravel(out, output.shape)];
  }
  if (relation.rule === "index") {
    const out = Array(output.shape.length).fill(0);
    for (const [axis, item] of relation.axes.entries()) {
      if (item.kind === "int") {
        if (coords[axis] !== item.index) return [];
        continue;
      }
      const delta = coords[axis] - item.start;
      if (delta < 0 || delta % item.step) return [];
      const position = delta / item.step;
      if (position >= output.shape[item.out]) return [];
      out[item.out] = position;
    }
    return [ravel(out, output.shape)];
  }
  // tile and elementwise: keep the current output position on reused axes.
  const out = unravel(current, output.shape);
  const offset = output.shape.length - input.shape.length;
  input.shape.forEach((size, axis) => {
    const at = offset + axis;
    if (relation.rule === "tile")
      out[at] = out[at] - (out[at] % size) + coords[axis];
    else if (size !== 1) out[at] = coords[axis];
  });
  return [ravel(out, output.shape)];
}

/** tril/triu keep a cell by position alone. */
export function triangleKeeps(
  kind: string,
  diagonal: number,
  coords: number[],
): boolean {
  const offset = coords.at(-1)! - coords.at(-2)!;
  return kind === "tril" ? offset <= diagonal : offset >= diagonal;
}

/**
 * Each sample's part of a class-index loss: the score at its target class and
 * −log p(target). Arithmetic uses at most 4096 recorded score values across
 * the preview. Missing, paged, or larger score rows retain their relationships
 * without allocating or traversing the tensor's full class dimension.
 */
export function classLossSamples(
  relation: Extract<Relation, { rule: "class_loss" }>,
  scores: Tensor,
  target: Tensor,
  selected: number,
  limit = 256,
) {
  const classes = scores.shape.at(-1)!;
  const count = scores.shape.length === 2 ? scores.shape[0] : 1;
  const samples =
    relation.reduction === "none" && count > 1
      ? [selected]
      : Array.from({ length: Math.min(count, limit) }, (_, i) => i);
  let remainingValues = 4096;
  return samples.map((sample) => {
    const label = target.values[sample];
    const known =
      typeof label === "number" &&
      Number.isInteger(label) &&
      label >= 0 &&
      label < classes;
    const ignored = label === relation.ignore_index;
    let loss: number | null = null;
    if (known && !ignored) {
      const offset = sample * classes;
      const value = scores.values[offset + label];
      if (relation.normalized) {
        if (typeof value === "number" && Number.isFinite(value)) loss = -value;
      } else if (
        classes <= remainingValues &&
        offset + classes <= scores.values.length
      ) {
        remainingValues -= classes;
        let top = -Infinity;
        let finite = true;
        for (let c = 0; c < classes; c++) {
          const score = scores.values[offset + c];
          if (typeof score !== "number" || !Number.isFinite(score)) {
            finite = false;
            break;
          }
          top = Math.max(top, score);
        }
        if (finite && typeof value === "number") {
          let total = 0;
          for (let c = 0; c < classes; c++)
            total += Math.exp((scores.values[offset + c] as number) - top);
          loss = -(value - top - Math.log(total));
        }
      }
    }
    return {
      sample,
      label: typeof label === "number" ? label : null,
      ignored,
      /** Flat index of the target-class score, when the label is valid. */
      cell: known ? sample * classes + label : null,
      loss,
    };
  });
}

export type AlignedAxis = {
  /** Null where the operand has no axis at this position. */
  size: number | null;
  /** True when this operand's single value is reused along the output axis. */
  stretched: boolean;
};

/**
 * Shapes lined up from the last axis, the way broadcasting compares them.
 * Returns null when nothing is stretched, so there is nothing to explain.
 */
export function alignShapes(
  shapes: number[][],
  output: number[],
): AlignedAxis[][] | null {
  const rows = shapes.map((shape) => {
    const offset = output.length - shape.length;
    return output.map((size, axis) => {
      const own = axis >= offset ? shape[axis - offset] : null;
      return { size: own, stretched: own !== size && size !== 1 };
    });
  });
  return rows.some((row) => row.some((axis) => axis.stretched)) ? rows : null;
}

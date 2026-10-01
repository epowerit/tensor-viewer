import type { Operation, Run, Tensor } from "../api/client";
import { tensorRelation } from "../operations/relations";
import { producedTensorIds } from "./provenance";

/**
 * Where each axis of a tensor comes from, traced back through the recorded
 * operations to the axes of inputs and parameters.
 *
 * A term names one source axis. An axis built by merging is a product of
 * terms; an axis cut out of a larger one is a part of a term. When an
 * operation computes an axis in a way that has no axis-level story (a
 * convolution's spatial output, a value-dependent selection), the lineage
 * stops there and says so rather than guessing.
 */
export type AxisTerm = {
  /** `x.features`, or `x axis 2` when the axis has no meaningful name. */
  label: string;
  size: number;
  /** For a piece of a split axis: which piece, out of how many. */
  part?: { index: number; of: number; sizes: number[] };
};
export type AxisStory = {
  size: number;
  terms: AxisTerm[];
  /** Why the trace stops, or what happened to an axis without a source. */
  note: string | null;
};

const named = (name: string) => !/^axis \d+$/.test(name);

function root(tensor: Tensor): AxisStory[] {
  return tensor.shape.map((size, axis) => ({
    size,
    terms: [
      {
        label: named(tensor.axes[axis] ?? "")
          ? `${tensor.name}.${tensor.axes[axis]}`
          : `${tensor.name} axis ${axis}`,
        size,
      },
    ],
    note: null,
  }));
}
const fresh = (size: number, note: string): AxisStory => ({
  size,
  terms: [],
  note,
});
const carry = (story: AxisStory, size = story.size): AxisStory =>
  size === story.size ? story : { ...story, size };

/** Consecutive pieces that rebuild a whole split axis become that axis again. */
function rejoin(terms: AxisTerm[]): AxisTerm[] {
  const first = terms[0]?.part;
  if (
    first &&
    terms.length === first.of &&
    terms.every(
      (term, index) =>
        term.label === terms[0].label &&
        term.part?.index === index &&
        term.part.of === first.of &&
        term.part.sizes.join() === first.sizes.join(),
    )
  ) {
    const { part: _, ...whole } = terms[0];
    return [{ ...whole, size: first.sizes.reduce((a, b) => a * b, 1) }];
  }
  return terms;
}

/**
 * Regroup axes the way reshape does: axes merge or split only within runs
 * whose element counts agree. Size-one axes carry no elements and are matched
 * separately.
 */
export function regroup(
  input: number[],
  output: number[],
  stories: AxisStory[],
): AxisStory[] {
  const inAxes = input.flatMap((size, axis) => (size === 1 ? [] : [axis]));
  const outAxes = output.flatMap((size, axis) => (size === 1 ? [] : [axis]));
  const result: AxisStory[] = output.map((size) =>
    fresh(size, "an axis of size 1"),
  );
  // Size-one axes hold no elements; carry them across only when their count
  // is unchanged, so their order identifies them.
  const inOnes = input.flatMap((size, axis) => (size === 1 ? [axis] : []));
  const outOnes = output.flatMap((size, axis) => (size === 1 ? [axis] : []));
  if (inOnes.length === outOnes.length)
    outOnes.forEach((axis, k) => (result[axis] = stories[inOnes[k]]));
  const failed = () =>
    output.map((size) => fresh(size, "regrouped by reshape"));
  let i = 0,
    j = 0;
  while (i < inAxes.length && j < outAxes.length) {
    const from = [inAxes[i++]],
      to = [outAxes[j++]];
    let left = input[from[0]],
      right = output[to[0]];
    while (left !== right) {
      if (left < right) {
        if (i >= inAxes.length) return failed();
        from.push(inAxes[i++]);
        left *= input[from.at(-1)!];
      } else {
        if (j >= outAxes.length) return failed();
        to.push(outAxes[j++]);
        right *= output[to.at(-1)!];
      }
    }
    const terms = rejoin(from.flatMap((axis) => stories[axis].terms));
    const unexplained = from.some((axis) => !stories[axis].terms.length);
    if (to.length === 1) {
      const story = from.length === 1 ? stories[from[0]] : null;
      result[to[0]] = story
        ? carry(story, output[to[0]])
        : unexplained
          ? fresh(output[to[0]], "merged by reshape")
          : { size: output[to[0]], terms, note: null };
    } else if (from.length === 1 && stories[from[0]].terms.length === 1) {
      const sizes = to.map((axis) => output[axis]);
      to.forEach((axis, index) => {
        result[axis] = {
          size: output[axis],
          terms: [
            {
              ...stories[from[0]].terms[0],
              size: output[axis],
              part: { index, of: to.length, sizes },
            },
          ],
          note: null,
        };
      });
    } else
      to.forEach(
        (axis) => (result[axis] = fresh(output[axis], "regrouped by reshape")),
      );
  }
  return result;
}

const REGROUP = new Set([
  "reshape",
  "view",
  "flatten",
  "unflatten",
  "squeeze",
  "unsqueeze",
  "ravel",
  "view_as",
  "reshape_as",
  "contiguous",
  "clone",
]);

/** Right-aligned operand axis that supplies an output axis of a given size. */
function aligned(
  operands: { shape: number[]; stories: AxisStory[] }[],
  rank: number,
  axis: number,
  size: number,
): AxisStory | null {
  for (const operand of operands) {
    const at = axis - (rank - operand.shape.length);
    if (at >= 0 && operand.shape[at] === size && size !== 1)
      return operand.stories[at];
  }
  return null;
}

/** Stories for the outputs of one operation, or null if it explains nothing. */
function through(
  op: Operation,
  outputIndex: number,
  output: Tensor,
  inputs: Tensor[],
  of: (tensor: Tensor) => AxisStory[],
): AxisStory[] | null {
  const lesson = op.lesson;
  const input = inputs[0];
  if (!input) return null;
  const rank = output.shape.length;
  const order = lesson.axis_order;
  if (
    order &&
    order.length === rank &&
    order.every((from, to) => input.shape[from] === output.shape[to])
  )
    return order.map((from) => of(input)[from]);
  if (lesson.mapping_rule === "identity" && REGROUP.has(op.kind))
    return regroup(input.shape, output.shape, of(input));
  if (lesson.mapping_rule === "roll") return of(input);

  const relation = tensorRelation(op, inputs, output);
  if (relation && outputIndex === 0) {
    const source = inputs[relation.operand];
    const stories = of(source);
    switch (relation.rule) {
      case "elementwise":
      case "prefix":
      case "channel_affine": {
        const operands = inputs.map((tensor) => ({
          shape: tensor.shape,
          stories: of(tensor),
        }));
        return output.shape.map(
          (size, axis) =>
            aligned(operands, rank, axis, size) ??
            fresh(size, size === 1 ? "an axis of size 1" : "broadcast"),
        );
      }
      case "pad":
        return stories.map((story, axis) =>
          output.shape[axis] === story.size
            ? story
            : { ...story, size: output.shape[axis], note: "padded" },
        );
      case "reduce": {
        const kept: AxisStory[] = [];
        stories.forEach((story, axis) => {
          if (!relation.axes.includes(axis)) kept.push(story);
          else if (relation.keepdim)
            kept.push(fresh(1, `reduced by ${op.kind}`));
        });
        return kept;
      }
      case "index": {
        const result = output.shape.map((size) =>
          fresh(size, "inserted by indexing"),
        );
        relation.axes.forEach((item, axis) => {
          if (item.kind === "slice")
            result[item.out] =
              output.shape[item.out] === stories[axis].size
                ? stories[axis]
                : {
                    ...stories[axis],
                    size: output.shape[item.out],
                    note: "sliced",
                  };
        });
        return result;
      }
      case "tile": {
        const offset = rank - source.shape.length;
        return output.shape.map((size, axis) => {
          if (axis < offset) return fresh(size, `new axis from ${op.kind}`);
          const story = stories[axis - offset];
          return size === story.size
            ? story
            : { ...story, size, note: `repeated ×${size / story.size}` };
        });
      }
      case "one_hot":
        return [
          ...stories,
          fresh(output.shape[rank - 1], "one position per class"),
        ];
      case "einsum": {
        const lettered = relation.inputs.map((letters, i) => ({
          letters,
          stories: of(inputs[i]),
        }));
        return [...relation.output].map((letter, axis) => {
          for (const operand of lettered) {
            const at = operand.letters.indexOf(letter);
            if (at >= 0) return operand.stories[at];
          }
          return fresh(output.shape[axis], `einsum letter ${letter}`);
        });
      }
      case "table":
        if (op.kind === "embedding" && inputs[1])
          return [...of(inputs[0]), of(inputs[1])[inputs[1].shape.length - 1]];
        return null;
      default:
        return null;
    }
  }
  if (lesson.interaction === "dot_product" && inputs[1] && rank >= 2) {
    const [left, right] = inputs;
    const operands = [left, right].map((tensor) => ({
      shape: tensor.shape.slice(0, -2),
      stories: of(tensor).slice(0, -2),
    }));
    return output.shape.map((size, axis) =>
      axis === rank - 2
        ? of(left)[left.shape.length - 2]
        : axis === rank - 1
          ? of(right)[right.shape.length - 1]
          : (aligned(operands, rank - 2, axis, size) ??
            fresh(size, "batch broadcast")),
    );
  }
  if (lesson.interaction === "linear_projection" && inputs[1]) {
    const weight = of(inputs[1]);
    return [...of(input).slice(0, -1), weight[0]];
  }
  if (
    ["normalization", "layer_normalization", "broadcast_add"].includes(
      lesson.interaction,
    )
  ) {
    const operands = inputs.map((tensor) => ({
      shape: tensor.shape,
      stories: of(tensor),
    }));
    return output.shape.map(
      (size, axis) =>
        aligned(operands, rank, axis, size) ?? fresh(size, "broadcast"),
    );
  }
  if (lesson.interaction === "tensor_assembly") {
    const raw = Number(op.arguments.dim ?? 0);
    if (op.kind === "stack") {
      const at = ((raw % rank) + rank) % rank;
      const stories = [...of(input)];
      stories.splice(at, 0, fresh(output.shape[at], "one per stacked tensor"));
      return stories;
    }
    if (["cat", "concat", "concatenate"].includes(op.kind)) {
      const at = ((raw % rank) + rank) % rank;
      return of(input).map((story, axis) =>
        axis === at
          ? {
              size: output.shape[axis],
              terms: [],
              note: `${inputs.length} parts joined by ${op.kind}`,
            }
          : story,
      );
    }
    if (input.shape.length === rank)
      return of(input).map((story, axis) =>
        output.shape[axis] === story.size
          ? story
          : {
              ...story,
              size: output.shape[axis],
              note: `part cut by ${op.kind}`,
            },
      );
  }
  return null;
}

/** Axis stories for one recorded tensor. */
export function axisLineage(
  trace: Run["trace"],
  tensorId: string,
): AxisStory[] {
  return lineageOf(trace)(tensorId);
}

/** A lookup that shares its work across every tensor of one trace. */
export function lineageOf(
  trace: Run["trace"],
): (tensorId: string) => AxisStory[] {
  const producers = new Map<string, { op: Operation; index: number }>();
  for (const op of trace.operations)
    producedTensorIds(op).forEach((id) => {
      if (!producers.has(id))
        producers.set(id, { op, index: op.outputs.indexOf(id) });
    });
  const memo = new Map<string, AxisStory[]>();
  const of = (tensor: Tensor): AxisStory[] => {
    const known = memo.get(tensor.id);
    if (known) return known;
    // A cycle cannot occur in a recorded trace, but guard against bad data.
    memo.set(tensor.id, root(tensor));
    const producer = producers.get(tensor.id);
    let stories = root(tensor);
    const inputs = (producer?.op.inputs ?? [])
      .map((id) => trace.tensors[id])
      .filter(Boolean);
    // A tensor created from nothing, like torch.ones(5, 8), is its own source.
    if (
      producer &&
      inputs.length &&
      producer.op.status === "ok" &&
      producer.index >= 0
    ) {
      const traced = through(producer.op, producer.index, tensor, inputs, of);
      stories =
        traced && traced.length === tensor.shape.length
          ? traced.map((story, axis) => carry(story, tensor.shape[axis]))
          : tensor.shape.map((size) =>
              fresh(size, `computed by ${producer.op.kind}`),
            );
    }
    memo.set(tensor.id, stories);
    return stories;
  };
  return (tensorId) => {
    const tensor = trace.tensors[tensorId];
    return tensor ? of(tensor) : [];
  };
}

/** One readable line per axis, such as "x.tokens × x.features". */
export function describeAxis(story: AxisStory): string {
  if (!story.terms.length) return story.note ?? "unknown";
  const text = story.terms
    .map((term) =>
      term.part
        ? `${term.label} (piece ${term.part.index + 1} of ${term.part.sizes.join("×")})`
        : term.label,
    )
    .join(" × ");
  return story.note ? `${text}, ${story.note}` : text;
}

/** True when every axis is simply the tensor's own: nothing to explain. */
export function isOwnLineage(tensor: Tensor, stories: AxisStory[]): boolean {
  const own = root(tensor);
  return stories.every(
    (story, axis) =>
      story.terms.length === 1 &&
      !story.terms[0].part &&
      !story.note &&
      story.terms[0].label === own[axis]?.terms[0].label,
  );
}

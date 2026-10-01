import { sourceIndex } from "../tensors/relationships";
import type { Operation, Tensor } from "../api/client";
import {
  dotContributors,
  formatValue,
  normalizationGroup,
  unravel,
} from "../tensors/coordinates";

export type Presentation = {
  leftHighlights: number[];
  rightHighlights: number[];
  title: string;
  text: string;
  expression?: string;
};
type Presenter = (
  op: Operation,
  inputs: Tensor[],
  output: Tensor,
  selected: number,
) => Presentation;

const inspect: Presenter = (_op, _inputs, output, selected) => ({
  leftHighlights: [],
  rightHighlights: [],
  title: "Inspect an element",
  text:
    output.values[selected] === undefined
      ? `Inspect [${unravel(selected, output.shape).join(", ")}]. ${output.value_source === "shape" ? "This run records shape and layout without numeric values." : "Values are loaded for the visible window."}`
      : `Output [${unravel(selected, output.shape).join(", ")}] = ${formatValue(output.values[selected])}. Select another cell to explore.`,
});

const mapping: Presenter = (op, inputs, output, selected) => {
  const index = inputs[0]
    ? sourceIndex(op, inputs[0], output, selected)
    : undefined;
  if (index === undefined || !inputs[0])
    return inspect(op, inputs, output, selected);
  return {
    leftHighlights: [index],
    rightHighlights: [],
    title: "Follow the same element",
    text: `${output.values[selected] === undefined ? "Follow the highlighted element." : `The highlighted value is ${formatValue(output.values[selected])}.`} ${JSON.stringify(unravel(index, inputs[0].shape)) === JSON.stringify(unravel(selected, output.shape)) ? "Its logical coordinates are unchanged at this position." : "Its coordinates change while its value stays the same."}`,
    expression: `[${unravel(index, inputs[0].shape).join(", ")}] → [${unravel(selected, output.shape).join(", ")}]`,
  };
};

const dotProduct: Presenter = (op, inputs, output, selected) => {
  if (inputs.length < 2) return inspect(op, inputs, output, selected);
  const pairs = dotContributors(
    inputs[0].shape,
    inputs[1].shape,
    output.shape,
    selected,
    256,
  );
  const terms = pairs
    .slice(0, 8)
    .map(
      (p) =>
        `${formatValue(inputs[0].values[p.left])} × ${formatValue(inputs[1].values[p.right])}`,
    );
  return {
    leftHighlights: pairs.map((p) => p.left),
    rightHighlights: pairs.map((p) => p.right),
    title: "One cell, one dot product",
    text: `Multiply the ${inputs[0].shape.at(-1)} matching entries in the highlighted row and column, then add. Up to 256 contributors are highlighted. Numeric terms appear when available.`,
    expression:
      pairs.some(
        (p) =>
          inputs[0].values[p.left] === undefined ||
          inputs[1].values[p.right] === undefined,
      ) || output.values[selected] === undefined
        ? undefined
        : `${terms.map((t) => `(${t})`).join(" + ") || "0"}${pairs.length > 8 ? " + …" : ""} ≈ ${formatValue(output.values[selected])}`,
  };
};

const normalization: Presenter = (op, inputs, output, selected) => {
  if (!inputs[0]) return inspect(op, inputs, output, selected);
  const dim = op.arguments.dim;
  const dtype = inputs[0].dtype.replace(/^torch\./, "");
  // Tensor details also serves older traces. Do not invent an axis or apply
  // arithmetic to pre-cast values; scalar formulas live in the full lesson.
  if (
    !output.shape.length ||
    typeof dim !== "number" ||
    !Number.isInteger(dim) ||
    dim < -output.shape.length ||
    dim >= output.shape.length ||
    inputs[0].shape.join() !== output.shape.join() ||
    !["float16", "bfloat16", "float32", "float64"].includes(dtype) ||
    output.dtype.replace(/^torch\./, "") !== dtype ||
    (op.arguments.dtype != null && op.arguments.dtype !== `torch.${dtype}`)
  )
    return inspect(op, inputs, output, selected);
  const group = normalizationGroup(
    output.shape,
    selected,
    Number(op.arguments.dim ?? -1),
    4096,
  );
  if (
    group.some((i) => inputs[0].values[i] === undefined) ||
    group.length <
      output.shape[
        (Number(op.arguments.dim ?? -1) + output.shape.length) %
          output.shape.length
      ]
  )
    return {
      leftHighlights: group.slice(0, 256),
      rightHighlights: [],
      title: "Normalize a group of scores",
      text: "Scores along this axis share a denominator. Up to 256 positions are highlighted; numeric calculations need the complete group.",
    };
  const values = group.map((i) => Number(inputs[0].values[i]));
  if (values.some((value) => !Number.isFinite(value)))
    return {
      leftHighlights: group,
      rightHighlights: [],
      title: "Inspect non-finite scores",
      text: "This group contains non-finite values. Inspect the recorded output; the usual finite-score normalization calculation does not apply.",
    };
  const max = Math.max(...values);
  const denominator = values.reduce((s, v) => s + Math.exp(v - max), 0);
  return {
    leftHighlights: group,
    rightHighlights: [],
    title: "Normalize a group of scores",
    text: `These ${group.length} scores share one denominator. Their output weights sum to one (up to floating-point rounding).`,
    expression: `exp(${formatValue(inputs[0].values[selected])} − ${formatValue(max)}) / ${formatValue(denominator)} ≈ ${formatValue(output.values[selected])}`,
  };
};

/** Reusable presenters consume trace data; none executes the user's algorithm. */
export const presenters: Record<string, Presenter> = {
  mapping,
  dot_product: dotProduct,
  normalization,
  inspect,
};

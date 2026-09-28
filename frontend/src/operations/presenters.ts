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
  text: `Output [${unravel(selected, output.shape).join(", ")}] = ${formatValue(output.values[selected])}. Select another cell to explore.`,
});

const mapping: Presenter = (op, inputs, output, selected) => {
  const index = op.lesson.mapping?.[selected];
  if (index === undefined || !inputs[0])
    return inspect(op, inputs, output, selected);
  return {
    leftHighlights: [index],
    rightHighlights: [],
    title: "Follow the same element",
    text: `The highlighted value is ${formatValue(output.values[selected])}. Its coordinates change while its value stays the same.`,
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
    text: `Multiply the ${pairs.length} matching entries in the highlighted row and column, then add. Displayed terms are rounded; the result comes from PyTorch.`,
    expression: `${terms.map((t) => `(${t})`).join(" + ")}${pairs.length > 8 ? " + …" : ""} = ${formatValue(output.values[selected])}`,
  };
};

const normalization: Presenter = (op, inputs, output, selected) => {
  if (!inputs[0]) return inspect(op, inputs, output, selected);
  const group = normalizationGroup(
    output.shape,
    selected,
    Number(op.arguments.dim ?? -1),
  );
  const values = group.map((i) => Number(inputs[0].values[i]));
  const max = Math.max(...values);
  const denominator = values.reduce((s, v) => s + Math.exp(v - max), 0);
  return {
    leftHighlights: group,
    rightHighlights: [],
    title: "Normalize a group of scores",
    text: `These ${group.length} scores share one denominator. Their output weights sum to one (up to floating-point rounding).`,
    expression: `exp(${formatValue(inputs[0].values[selected])} − ${formatValue(max)}) / ${formatValue(denominator)} = ${formatValue(output.values[selected])}`,
  };
};

/** Reusable presenters consume trace data; none executes the user's algorithm. */
export const presenters: Record<string, Presenter> = {
  mapping,
  dot_product: dotProduct,
  normalization,
  inspect,
};

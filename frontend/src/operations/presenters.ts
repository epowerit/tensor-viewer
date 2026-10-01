import { sourceIndex } from "../tensors/relationships";
import {
  broadcastIndex,
  classLossSamples,
  einsumTerms,
  prefixGroup,
  reductionGroup,
  relationSource,
  tensorRelation,
  triangleKeeps,
} from "./relations";
import type { Operation, Tensor } from "../api/client";
import {
  dotContributors,
  formatValue,
  normalizationGroup,
  unravel,
} from "../tensors/coordinates";

export type Presentation = {
  /** Highlights per operand position, when more than the first takes part. */
  operandHighlights?: number[][];
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
    text:
      op.kind === "log_softmax"
        ? `These ${group.length} scores share one denominator. The exponentials of their outputs sum to one (up to floating-point rounding).`
        : `These ${group.length} scores share one denominator. Their output weights sum to one (up to floating-point rounding).`,
    expression:
      op.kind === "log_softmax"
        ? `(${formatValue(inputs[0].values[selected])} − ${formatValue(max)}) − log(${formatValue(denominator)}) ≈ ${formatValue(output.values[selected])}`
        : `exp(${formatValue(inputs[0].values[selected])} − ${formatValue(max)}) / ${formatValue(denominator)} ≈ ${formatValue(output.values[selected])}`,
  };
};

const SYMBOLS: Record<string, string> = {
  add: "+",
  sub: "−",
  rsub: "−",
  __rsub__: "−",
  mul: "×",
  div: "÷",
  true_divide: "÷",
  __rdiv__: "÷",
  __rtruediv__: "÷",
  floor_divide: "//",
  remainder: "%",
  pow: "^",
  __rpow__: "^",
  eq: "==",
  ne: "≠",
  gt: ">",
  ge: "≥",
  lt: "<",
  le: "≤",
  logical_and: "and",
  __and__: "&",
  logical_or: "or",
  __or__: "|",
  logical_xor: "xor",
  __xor__: "xor",
};
const argumentValue = (value: unknown): string | undefined =>
  typeof value === "boolean"
    ? value
      ? "True"
      : "False"
    : typeof value === "number"
      ? String(value)
      : typeof value === "string" && value !== "tensor"
        ? JSON.stringify(value)
        : undefined;
const at = (tensor: Tensor, index: number) =>
  `[${unravel(index, tensor.shape).join(", ")}]`;
const shown = (tensor: Tensor, value: number | string | undefined) =>
  value === undefined
    ? undefined
    : tensor.dtype === "bool"
      ? Number(value)
        ? "True"
        : "False"
      : formatValue(value);

/** Rules recorded for selection, repetition, reduction, and elementwise work. */
const relation: Presenter = (op, inputs, output, selected) => {
  const rule = tensorRelation(op, inputs, output);
  if (!rule) return inspect(op, inputs, output, selected);
  const result = shown(output, output.values[selected]);
  const highlights: number[][] = inputs.map(() => []);
  const base = { rightHighlights: [] as number[] };
  if (rule.rule === "reduce") {
    const input = inputs[rule.operand];
    const group = reductionGroup(rule, input, output, selected);
    highlights[rule.operand] = group.indices;
    const values = group.indices.map((i) => input.values[i]);
    const complete =
      group.size === group.indices.length &&
      values.every((v) => v !== undefined) &&
      result !== undefined;
    const joiner =
      op.kind === "prod"
        ? " × "
        : op.kind === "sum" || op.kind === "mean"
          ? " + "
          : ", ";
    const terms = values
      .slice(0, 8)
      .map((v) => formatValue(v))
      .join(joiner);
    const body = `${terms}${group.size > 8 ? `${joiner}…` : ""}`;
    return {
      ...base,
      operandHighlights: highlights,
      leftHighlights: highlights[0],
      title: "One output cell, one group",
      text: `${op.kind} combines the ${group.size.toLocaleString()} input cells that differ only along axis ${rule.axes.join(", ")}.${group.size > group.indices.length ? ` The first ${group.indices.length} are highlighted.` : ""}`,
      expression: complete
        ? op.kind === "sum" || op.kind === "prod"
          ? `${body} = ${result}`
          : op.kind === "mean"
            ? `(${body}) ÷ ${group.size} = ${result}`
            : `${op.kind}(${body}) = ${result}`
        : undefined,
    };
  }
  if (rule.rule === "prefix") {
    const input = inputs[rule.operand];
    const group = prefixGroup(rule, input, selected);
    highlights[rule.operand] = group.indices;
    const values = group.indices.map((i) => input.values[i]);
    const joiner = op.kind === "cumprod" ? " × " : " + ";
    const complete =
      group.size === group.indices.length &&
      values.every((v) => v !== undefined) &&
      result !== undefined;
    const shownTerms = values.slice(-8).map((v) => formatValue(v));
    return {
      ...base,
      operandHighlights: highlights,
      leftHighlights: highlights[0],
      title: "A running total",
      text:
        group.size === 1
          ? `This is the first position along axis ${rule.axis}, so it holds the input cell itself.`
          : `This cell accumulates the ${group.size.toLocaleString()} input cells from the start of axis ${rule.axis} up to its own position.`,
      expression: complete
        ? `${group.size > 8 ? `…${joiner}` : ""}${shownTerms.join(joiner)} = ${result}`
        : undefined,
    };
  }
  if (rule.rule === "einsum") {
    const { terms, size } = einsumTerms(
      rule,
      inputs.map((tensor) => tensor.shape),
      output.shape,
      selected,
    );
    inputs.forEach(
      (_, operand) =>
        (highlights[operand] = [
          ...new Set(terms.map((term) => term[operand])),
        ]),
    );
    const factors = terms.map((term) =>
      term.map((index, operand) => inputs[operand].values[index]),
    );
    const complete =
      size === terms.length &&
      result !== undefined &&
      factors.every((term) => term.every((v) => v !== undefined));
    const summed = [...new Set(rule.inputs.join(""))].filter(
      (letter) => !rule.output.includes(letter),
    );
    const written = factors
      .slice(0, 6)
      .map((term) => term.map((v) => formatValue(v)).join(" × "));
    return {
      ...base,
      operandHighlights: highlights,
      leftHighlights: highlights[0],
      title:
        size === 1
          ? "One product"
          : `A sum of ${size.toLocaleString()} products`,
      text: `${rule.inputs.join(", ")} → ${rule.output || "scalar"}. The output cell fixes ${rule.output ? [...rule.output].join(", ") : "no letters"}; ${summed.length ? `${summed.join(", ")} ${summed.length === 1 ? "is" : "are"} summed over` : "nothing is summed"}.${size > terms.length ? ` The first ${terms.length} products are highlighted.` : ""}`,
      expression: complete
        ? `${written.map((t) => (size > 1 ? `(${t})` : t)).join(" + ")}${size > 6 ? " + …" : ""} = ${result}`
        : undefined,
    };
  }
  if (rule.rule === "channel_affine") {
    const channel = unravel(selected, output.shape)[rule.axis];
    inputs.forEach(
      (_, operand) => (highlights[operand] = [operand ? channel : selected]),
    );
    const number = (role: string) => {
      const operand = rule.roles.indexOf(role);
      const value =
        operand < 0
          ? undefined
          : inputs[operand].values[operand ? channel : selected];
      return typeof value === "number" ? value : undefined;
    };
    const x = number("input"),
      mean = number("running_mean"),
      variance = number("running_var");
    const scale = number("weight"),
      shift = number("bias");
    const affine =
      (rule.roles.includes("weight") ? ` × ${formatValue(scale)}` : "") +
      (rule.roles.includes("bias") ? ` + ${formatValue(shift)}` : "");
    const complete =
      x !== undefined &&
      mean !== undefined &&
      variance !== undefined &&
      result !== undefined &&
      (!rule.roles.includes("weight") || scale !== undefined) &&
      (!rule.roles.includes("bias") || shift !== undefined);
    return {
      ...base,
      operandHighlights: highlights,
      leftHighlights: highlights[0],
      title: `Channel ${channel}'s stored statistics`,
      text: `Every cell of channel ${channel} uses the same stored mean and variance${rule.roles.length > 3 ? ", then the channel's scale and shift" : ""}. Nothing else in the batch affects this value.`,
      expression: complete
        ? `(${formatValue(x)} − ${formatValue(mean)}) ÷ √(${formatValue(variance)} + ${rule.eps})${affine} ≈ ${result}`
        : undefined,
    };
  }
  if (rule.rule === "one_hot") {
    const index = inputs[rule.operand];
    const classes = output.shape.at(-1)!;
    const source = Math.floor(selected / classes);
    const position = selected % classes;
    highlights[rule.operand] = [source];
    const value = index.values[source];
    return {
      ...base,
      operandHighlights: highlights,
      leftHighlights: highlights[0],
      title:
        value === undefined
          ? "One position per class"
          : value === position
            ? "This is the named class: 1"
            : "Another class: 0",
      text: `The last axis has one position per class. This cell is position ${position}; it is 1 only when the highlighted index equals ${position}.`,
      expression:
        value === undefined || result === undefined
          ? undefined
          : `${formatValue(value)} == ${position} → ${result}`,
    };
  }
  if (rule.rule === "class_loss") {
    const [scores, target] = inputs;
    const samples = classLossSamples(rule, scores, target, selected);
    const counted = samples.filter((item) => !item.ignored);
    const total = scores.shape.length === 2 ? scores.shape[0] : 1;
    const allTargetsKnown =
      target.values.length === target.numel &&
      target.values.every(
        (value) => typeof value === "number" && Number.isInteger(value),
      );
    const contributing = allTargetsKnown
      ? target.values.filter((value) => value !== rule.ignore_index).length
      : null;
    const reduced = rule.reduction !== "none";
    const preview = reduced && samples.length < total;
    const noContributors = reduced && contributing === 0;
    const ignoredSample = !reduced && samples[0]?.ignored;
    highlights[0] = counted.flatMap((item) =>
      item.cell === null ? [] : [item.cell],
    );
    highlights[1] = samples.map((item) => item.sample);
    const complete =
      result !== undefined &&
      counted.length > 0 &&
      counted.every((item) => item.loss !== null) &&
      samples.length ===
        (scores.shape.length === 2 && rule.reduction !== "none"
          ? scores.shape[0]
          : 1);
    const terms = counted
      .slice(0, 6)
      .map((item) => formatValue(item.loss ?? undefined));
    const more = counted.length > 6 ? ", …" : "";
    const one = samples.length === 1 ? counted[0] : undefined;
    return {
      ...base,
      operandHighlights: highlights,
      leftHighlights: highlights[0],
      title: noContributors
        ? "No contributing samples"
        : ignoredSample
          ? "An ignored target"
          : !reduced || total === 1
            ? "One sample, one target class"
            : `${rule.reduction === "sum" ? "The sum" : "The average"} over ${contributing === null ? "non-ignored targets" : `${contributing.toLocaleString()} samples`}`,
      text: `Each non-ignored sample contributes −log p(target): the ${rule.normalized ? "log-probability" : "score"} highlighted in its row${rule.normalized ? "" : ", after the row is turned into log-probabilities"}. A confident correct score gives a loss near zero.${preview ? ` Showing the first ${samples.length.toLocaleString()} of ${total.toLocaleString()} samples; the recorded result includes the entire batch.` : ""}${contributing === null ? " The number of contributing samples is unknown until their target values are available." : contributing < total ? " Targets equal to the ignore index are skipped." : ""}`,
      expression:
        (noContributors || ignoredSample) && result !== undefined
          ? `${noContributors ? "All targets ignored" : "Ignored target"} → ${result}`
          : !complete
            ? undefined
            : one
              ? `−log p(class ${one.label}) = ${formatValue(one.loss ?? undefined)}${rule.reduction === "none" || counted.length === 1 ? "" : ` → ${result}`}`
              : `${rule.reduction === "sum" ? "sum" : "mean"}(${terms.join(", ")}${more}) = ${result}`,
    };
  }
  if (rule.rule === "pad") {
    const input = inputs[rule.operand];
    const source = relationSource(rule, input, output, selected);
    if (source !== undefined) highlights[rule.operand] = [source];
    return {
      ...base,
      operandHighlights: highlights,
      leftHighlights: highlights[0],
      title: source === undefined ? "A border cell" : "An original cell",
      text:
        source === undefined
          ? "This cell lies in the added border. It holds the fill value and has no input cell behind it."
          : "This cell is the highlighted input cell, shifted by the padding before it on each axis.",
      expression:
        source === undefined
          ? result === undefined
            ? undefined
            : `fill = ${result}`
          : `${input.name}${at(input, source)} → ${at(output, selected)}${result === undefined ? "" : ` = ${result}`}`,
    };
  }
  if (rule.rule !== "elementwise") {
    const input = inputs[rule.operand];
    const source = relationSource(rule, input, output, selected);
    if (source === undefined) return inspect(op, inputs, output, selected);
    highlights[rule.operand] = [source];
    return {
      ...base,
      operandHighlights: highlights,
      leftHighlights: highlights[0],
      title:
        rule.rule === "index"
          ? "Follow the selected cell"
          : rule.rule === "tile"
            ? "Many outputs, one stored value"
            : "Follow the looked-up cell",
      text:
        rule.rule === "index"
          ? "This output cell is the highlighted input cell. Integers fix a position, and slices step along an axis."
          : rule.rule === "tile"
            ? "This output cell reads the highlighted input cell. Other outputs along the repeated axes read the same one."
            : `This output cell copies the highlighted cell of ${input.name}, chosen by the recorded index values.`,
      expression: `${input.name}${at(input, source)} → ${at(output, selected)}${result === undefined ? "" : ` = ${result}`}`,
    };
  }
  const cells = inputs.map((tensor) =>
    broadcastIndex(tensor.shape, output.shape, selected),
  );
  cells.forEach((index, i) => (highlights[i] = [index]));
  const value = (role: string) => {
    const position = rule.roles.indexOf(role);
    if (position >= 0)
      return shown(inputs[position], inputs[position].values[cells[position]]);
    const scalar = op.arguments[role];
    return typeof scalar === "boolean"
      ? scalar
        ? "True"
        : "False"
      : typeof scalar === "number" || typeof scalar === "string"
        ? formatValue(scalar)
        : undefined;
  };
  const truthy = (role: string) => {
    const position = rule.roles.indexOf(role);
    const raw = inputs[position]?.values[cells[position]];
    return raw === undefined ? undefined : Number(raw) !== 0;
  };
  const coords = unravel(selected, output.shape);
  const reused = inputs.some((tensor) => tensor.numel < output.numel);
  let title = "Same position in every operand";
  let text = `Each output cell uses the cells at the matching position.${reused ? " A smaller operand is reused along the axes it lacks or where its size is one." : ""}`;
  let expression: string | undefined;
  const a = value("input"),
    b = value("other");
  const parameters = Object.entries(op.arguments)
    .filter(([key]) => !["input", "other", "out"].includes(key))
    .flatMap(([key, value]) => {
      const text = argumentValue(value);
      return text === undefined ? [] : [{ key, text }];
    });
  const call = () =>
    `${op.kind}(${[a, b, ...parameters.map(({ key, text }) => `${key}=${text}`)]
      .filter((value) => value !== undefined)
      .join(", ")}) = ${result}`;
  if (rule.diagonal !== undefined) {
    const kept = triangleKeeps(op.kind, rule.diagonal, coords);
    title = kept ? "Inside the triangle: kept" : "Outside the triangle: zero";
    text = `Row ${coords.at(-2)}, column ${coords.at(-1)}. ${op.kind} keeps a cell when column − row is ${op.kind === "tril" ? "at most" : "at least"} ${rule.diagonal}.`;
    expression = `${coords.at(-1)} − ${coords.at(-2)} = ${coords.at(-1)! - coords.at(-2)!} → ${kept ? `keep ${a ?? "the value"}` : "0"}`;
  } else if (op.kind === "where") {
    const condition = truthy("condition");
    title = "The condition chooses the source";
    text = "True takes the first value; false takes the second.";
    if (condition !== undefined && result !== undefined)
      expression = `condition is ${condition ? "True" : "False"} → ${condition ? "first" : "second"} value${(condition ? a : b) === undefined ? "" : ` ${condition ? a : b}`} = ${result}`;
  } else if (op.kind === "masked_fill") {
    const masked = truthy("mask");
    title = "The mask decides";
    text =
      "True cells are overwritten with the fill value; false cells keep the input.";
    if (masked !== undefined && result !== undefined)
      expression = `mask is ${masked ? "True" : "False"} → ${masked ? `filled with ${value("value") ?? "the fill value"}` : `keeps ${a ?? "the input"}`} = ${result}`;
  } else if (
    SYMBOLS[op.kind] &&
    a !== undefined &&
    b !== undefined &&
    result !== undefined
  ) {
    const [left, right] = rule.reversed || op.kind === "rsub" ? [b, a] : [a, b];
    const handled = new Set<string>();
    let scaled = right;
    if (["add", "sub", "rsub"].includes(op.kind)) {
      handled.add("alpha");
      const alpha = op.arguments.alpha;
      if (alpha !== undefined && alpha !== 1)
        scaled = `${argumentValue(alpha)} × ${right}`;
    }
    let body = `${left} ${SYMBOLS[op.kind]} ${scaled}`;
    if (["div", "__rdiv__", "__rtruediv__"].includes(op.kind)) {
      const rounding = op.arguments.rounding_mode;
      if (rounding === "floor" || rounding === "trunc") {
        handled.add("rounding_mode");
        body = `${rounding}(${body})`;
      }
    }
    expression = parameters.some(({ key }) => !handled.has(key))
      ? call()
      : `${body} = ${result}`;
  } else if (a !== undefined && result !== undefined) expression = call();
  return {
    ...base,
    operandHighlights: highlights,
    leftHighlights: highlights[0],
    title,
    text,
    expression,
  };
};

/** Reusable presenters consume trace data; none executes the user's algorithm. */
export const presenters: Record<string, Presenter> = {
  mapping,
  relation,
  dot_product: dotProduct,
  normalization,
  inspect,
};

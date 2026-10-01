import type { Draft, Operation, Run, Tensor } from "../api/client";
import { isCodeAt } from "../editor/completions";
import { entryPath, sourceCode } from "../sources/files";

export type ConsoleExample = {
  id: string;
  title: string;
  description: string;
  script: string;
  input: Pick<Draft["input"], "shape" | "axis_names" | "generator"> &
    Partial<Pick<Draft["input"], "dtype" | "text">>;
};

/** An example replaces the value source but keeps seeds and other settings. */
export function exampleInput(
  current: Draft["input"],
  example: ConsoleExample,
): Draft["input"] {
  return {
    ...current,
    uploaded: null,
    text: null,
    dtype: "float32",
    ...example.input,
  };
}

export const EXAMPLES: ConsoleExample[] = [
  {
    id: "layout",
    title: "Reshape and permute",
    description: "Regroup, reorder, and flatten the same 24 values.",
    input: {
      shape: [2, 3, 4],
      axis_names: ["batch", "rows", "columns"],
      generator: "arange",
    },
    script: `grouped = x.reshape(2, 3, 2, 2)
swapped = grouped.permute(0, 2, 1, 3)
packed = swapped.contiguous()
flat = packed.flatten(1)`,
  },
  {
    id: "broadcast",
    title: "Broadcasting",
    description: "Combine a column and a row into a grid.",
    input: {
      shape: [3, 4],
      axis_names: ["rows", "columns"],
      generator: "ones",
    },
    script: `row = torch.arange(4.0)
column = torch.arange(3.0).unsqueeze(1)
grid = column * 10 + row
shifted = x + grid`,
  },
  {
    id: "views",
    title: "Views and in-place writes",
    description: "See which tensors share storage when one is written.",
    input: { shape: [1, 2, 3], axis_names: [], generator: "arange" },
    script: `view = x.view(2, 3)
frozen = x.clone()
view[0, 1] = 99
x.add_(10)
return view, frozen, x`,
  },
  {
    id: "attention",
    title: "Attention scores",
    description: "Compare every token with every other token.",
    input: {
      shape: [1, 4, 8],
      axis_names: ["batch", "tokens", "features"],
      generator: "random",
    },
    script: `keys = x.transpose(-2, -1)
scores = x @ keys / math.sqrt(8)
weights = torch.softmax(scores, dim=-1)
mixed = weights @ x`,
  },
  {
    id: "convolution",
    title: "Convolution and pooling",
    description: "Find the edges of a picture, then shrink the result.",
    input: {
      shape: [1, 3, 16, 16],
      axis_names: ["batch", "channels", "height", "width"],
      generator: "image",
    },
    script: `kernel = torch.tensor([[-1.0, 0.0, 1.0]] * 3).expand(1, 3, 3, 3)
edges = F.conv2d(x, kernel, padding=1)
pooled = F.max_pool2d(edges, 2)`,
  },
  {
    id: "embedding",
    title: "Words to vectors",
    description: "Look up a vector for each token, then compare the tokens.",
    input: {
      shape: [1, 6],
      axis_names: ["batch", "tokens"],
      generator: "text",
      text: "the cat sat on the mat",
      dtype: "int64",
    },
    script: `table = torch.randn(5, 4)
vectors = F.embedding(x, table)
scores = vectors @ vectors.transpose(-2, -1)
weights = torch.softmax(scores, dim=-1)`,
  },
  {
    id: "assembly",
    title: "Join and split",
    description: "Concatenate, stack, and take tensors apart again.",
    input: { shape: [2, 3], axis_names: [], generator: "arange" },
    script: `joined = torch.cat([x, x + 100], dim=0)
stacked = torch.stack([x, x + 100], dim=0)
first, second = joined.chunk(2, dim=0)`,
  },
  {
    id: "selection",
    title: "Masks, sorting, and running totals",
    description: "Pick values by condition, order them, and accumulate.",
    input: { shape: [3, 4], axis_names: [], generator: "random" },
    script: `mask = x > 0
positive = x[mask]
ordered, positions = x.sort(dim=-1, descending=True)
best = x.topk(2, dim=-1)
running = ordered.cumsum(-1)
clipped = torch.where(mask, x, 0.0)`,
  },
  {
    id: "einsum",
    title: "Einstein sums",
    description: "Write products by naming each axis with a letter.",
    input: { shape: [2, 3, 4], axis_names: [], generator: "arange" },
    script: `pairs = torch.einsum("bij,bkj->bik", x, x)
totals = torch.einsum("bij->bi", x)
swapped = torch.einsum("bij->bji", x)
outer = torch.outer(x[0, 0], x[0, 1])`,
  },
  {
    id: "loss",
    title: "Classification loss",
    description: "From class numbers and scores to one number.",
    input: {
      shape: [3, 4],
      axis_names: ["samples", "classes"],
      generator: "random",
    },
    script: `labels = torch.tensor([2, 0, 3])
targets = F.one_hot(labels, 4)
log_probabilities = F.log_softmax(x, dim=-1)
per_sample = F.nll_loss(log_probabilities, labels, reduction="none")
loss = F.cross_entropy(x, labels)`,
  },
];

export function consoleProject(
  name: string,
  example: ConsoleExample = EXAMPLES[0],
): Draft {
  return {
    name,
    // The backend derives the module from the script on every save.
    code: "pass\n",
    script: example.script,
    class_name: "Console",
    constructor: {},
    capture_mode: "values",
    blueprint: null,
    input: exampleInput(
      {
        shape: [],
        axis_names: [],
        generator: "arange",
        dtype: "float32",
        seed: 7,
        random_stream: "input",
      },
      example,
    ),
  };
}

export type LineResult = {
  operations: Operation[];
  output: Tensor | null;
  error: string | null;
  /** False when this line, an earlier line, or an input changed after the run. */
  fresh: boolean;
  /** From a shapes-only check of the current code, not from a recorded run. */
  predicted?: boolean;
  /** The check could not go past this line because its shape depends on values. */
  needsValues?: boolean;
};

/**
 * A shape check has no values, so a step whose result shape depends on them
 * (a Boolean mask, nonzero, a data-dependent branch) stops it. That is a limit
 * of the check, not a mistake in the code.
 */
export function needsValues(
  error: { type: string; message: string } | null | undefined,
): boolean {
  return (
    !!error &&
    (error.type === "NotImplementedError" ||
      /meta tensor|data[- ]dependent|\bmeta\b.*not (supported|implemented)/i.test(
        error.message,
      ))
  );
}

/**
 * Fill the lines a recorded run no longer covers with a shape check of the
 * current code. Recorded results win wherever they are still current; a
 * predicted line has a shape but no step to open and no values.
 */
export function withShapeCheck(
  recorded: Map<number, LineResult>,
  checked: Map<number, LineResult>,
): Map<number, LineResult> {
  const merged = new Map(recorded);
  for (const [line, result] of checked) {
    if (!result.fresh || recorded.get(line)?.fresh) continue;
    if (!result.output && !result.error) continue;
    const limited =
      !!result.error &&
      needsValues({
        type: result.error.split(":")[0],
        message: result.error,
      });
    merged.set(line, {
      operations: [],
      output: limited ? null : result.output,
      error: limited ? null : result.error,
      fresh: true,
      predicted: true,
      needsValues: limited,
    });
  }
  return merged;
}

/**
 * Recorded results for each line of one source file, keyed by 1-based line
 * number. `file` is a project-relative path; null means the entry file.
 */
export function lineResults(
  run: Run | null,
  text: string,
  inputsChanged = false,
  file: string | null = null,
): Map<number, LineResult> {
  const results = new Map<number, LineResult>();
  if (!run) return results;
  const entry = entryPath(run.project);
  const path = file ?? entry;
  const recorded = sourceCode(run.project, path).split("\n");
  const current = text.split("\n");
  let unchanged = 0;
  while (
    unchanged < recorded.length &&
    unchanged < current.length &&
    recorded[unchanged] === current[unchanged]
  )
    unchanged++;
  const at = (line: number) => {
    if (!results.has(line))
      results.set(line, {
        operations: [],
        output: null,
        error: null,
        fresh: !inputsChanged && line <= unchanged,
      });
    return results.get(line)!;
  };
  for (const op of run.trace.operations) {
    if (!op.source || (op.source.file ?? entry) !== path) continue;
    const result = at(op.source.line);
    result.operations.push(op);
    const produced =
      op.outputs.at(0) ?? op.mutations?.at(-1)?.after ?? undefined;
    if (produced) result.output = run.trace.tensors[produced] ?? result.output;
    if (op.error) result.error = op.error;
  }
  const error = run.trace.error;
  if (error?.line != null && (error.file ?? entry) === path)
    at(error.line).error = `${error.type}: ${error.message}`;
  return results;
}

/** The most recently assigned top-level name, or the input when there is none. */
export function lastVariable(script: string, fallback = "x"): string {
  const assignments = [
    ...script.matchAll(/^([A-Za-z_]\w*)[ \t]*(?::[^=\n]+)?=(?!=)/gm),
  ];
  for (const match of assignments.reverse())
    if (isCodeAt(script, match.index)) return match[1];
  return fallback;
}

export type Snippet = {
  id: string;
  label: string;
  group: "Shape" | "Combine" | "Compute" | "Memory";
  name: string;
  /** `rank` is null when the operand's shape has not been recorded yet. */
  expression: (operand: string, shape: number[] | null) => string;
};

const lastAxes = (shape: number[] | null) =>
  shape && shape.length > 1
    ? shape
        .map((_, i) => i)
        .reverse()
        .join(", ")
    : "1, 0";

export const SNIPPETS: Snippet[] = [
  {
    id: "reshape",
    label: "reshape",
    group: "Shape",
    name: "reshaped",
    expression: (v, shape) =>
      `${v}.reshape(${shape?.length ? `-1, ${shape.at(-1)}` : "-1"})`,
  },
  {
    id: "permute",
    label: "permute",
    group: "Shape",
    name: "permuted",
    expression: (v, shape) => `${v}.permute(${lastAxes(shape)})`,
  },
  {
    id: "transpose",
    label: "transpose",
    group: "Shape",
    name: "transposed",
    expression: (v) => `${v}.transpose(-2, -1)`,
  },
  {
    id: "flatten",
    label: "flatten",
    group: "Shape",
    name: "flat",
    expression: (v, shape) =>
      `${v}.flatten(${shape && shape.length > 2 ? 1 : 0})`,
  },
  {
    id: "unsqueeze",
    label: "unsqueeze",
    group: "Shape",
    name: "expanded",
    expression: (v) => `${v}.unsqueeze(0)`,
  },
  {
    id: "squeeze",
    label: "squeeze",
    group: "Shape",
    name: "squeezed",
    expression: (v) => `${v}.squeeze()`,
  },
  {
    id: "roll",
    label: "roll",
    group: "Shape",
    name: "rolled",
    expression: (v) => `${v}.roll(1, dims=-1)`,
  },
  {
    id: "cat",
    label: "cat",
    group: "Combine",
    name: "joined",
    expression: (v) => `torch.cat([${v}, ${v}], dim=0)`,
  },
  {
    id: "stack",
    label: "stack",
    group: "Combine",
    name: "stacked",
    expression: (v) => `torch.stack([${v}, ${v}], dim=0)`,
  },
  {
    id: "add",
    label: "add",
    group: "Compute",
    name: "shifted",
    expression: (v) => `${v} + 1`,
  },
  {
    id: "matmul",
    label: "matmul",
    group: "Compute",
    name: "product",
    expression: (v) => `${v} @ ${v}.transpose(-2, -1)`,
  },
  {
    id: "sum",
    label: "sum",
    group: "Compute",
    name: "total",
    expression: (v) => `${v}.sum(dim=-1)`,
  },
  {
    id: "mean",
    label: "mean",
    group: "Compute",
    name: "average",
    expression: (v) => `${v}.mean(dim=-1, keepdim=True)`,
  },
  {
    id: "softmax",
    label: "softmax",
    group: "Compute",
    name: "weights",
    expression: (v) => `torch.softmax(${v}, dim=-1)`,
  },
  {
    id: "contiguous",
    label: "contiguous",
    group: "Memory",
    name: "packed",
    expression: (v) => `${v}.contiguous()`,
  },
  {
    id: "clone",
    label: "clone",
    group: "Memory",
    name: "copy",
    expression: (v) => `${v}.clone()`,
  },
];

/** Add an executable assignment, preserving an explicit top-level return. */
export function appendSnippet(
  script: string,
  snippet: Snippet,
  input: string,
  shapes: Record<string, number[]> = {},
): string {
  const body = script.replace(/\s+$/, "");
  // A top-level return ends the generated forward method. Insert before it
  // rather than adding unreachable code, including when its value spans lines.
  const returns = [...body.matchAll(/^return\b/gm)];
  const beforeReturn = returns.find((match) =>
    isCodeAt(body, match.index),
  )?.index;
  const operand = lastVariable(body.slice(0, beforeReturn), input);
  const used = new Set(script.match(/[A-Za-z_]\w*/g) ?? []);
  let name = snippet.name;
  for (let i = 2; used.has(name) || name === input; i++)
    name = `${snippet.name}${i}`;
  const line = `${name} = ${snippet.expression(operand, shapes[operand] ?? null)}`;
  if (beforeReturn !== undefined)
    return `${body.slice(0, beforeReturn)}${line}\n${body.slice(beforeReturn)}`;
  return body ? `${body}\n${line}` : line;
}

/** Shapes of named tensors whose recorded lines are still current. */
export function knownShapes(
  results: Map<number, LineResult>,
): Record<string, number[]> {
  const shapes: Record<string, number[]> = {};
  [...results.entries()]
    .sort(([a], [b]) => a - b)
    .forEach(([, result]) => {
      if (result.fresh && result.output)
        shapes[result.output.name] = result.output.shape;
    });
  return shapes;
}

/** Parse "2, 3, 4" into a shape, or explain why it is not one. */
export function parseShape(text: string): number[] | string {
  const parts = text
    .replace(/[[\]()]/g, "")
    .split(/[\s,x×]+/)
    .filter(Boolean);
  if (!parts.length) return "Enter at least one dimension, such as 2, 3, 4.";
  if (parts.length > 6) return "Use at most six dimensions.";
  const shape = parts.map(Number);
  if (shape.some((n) => !Number.isInteger(n) || n < 1))
    return "Dimensions are positive whole numbers.";
  return shape;
}

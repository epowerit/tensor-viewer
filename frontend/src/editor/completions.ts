export type Completion = {
  label: string;
  /** Text inserted in place of the typed prefix. */
  insert: string;
  /** The resulting shape when it is known, otherwise a short description. */
  detail: string;
};

export type CompletionRange = { from: number; to: number };

/** Insert a suggestion without keeping the rest of a partially edited name. */
export function applyCompletion(
  source: string,
  range: CompletionRange,
  item: Completion,
): { value: string; caret: number } {
  const suffix = source.slice(range.to);
  // Existing arguments belong to the user's code, not the suggestion.
  const insert = /^[ \t]*\(/.test(suffix) ? item.label : item.insert;
  return {
    value: source.slice(0, range.from) + insert + suffix,
    caret: range.from + insert.length,
  };
}

/** Keep suggestions out of comments and quoted text, including multiline strings. */
export function isCodeAt(source: string, caret: number): boolean {
  let quote: "'" | '"' | null = null;
  let triple = false;
  let comment = false;
  for (let i = 0; i < caret; i++) {
    const character = source[i];
    if (comment) {
      if (character === "\n") comment = false;
    } else if (quote) {
      if (character === "\\") i++;
      else if (
        triple ? source.startsWith(quote.repeat(3), i) : character === quote
      ) {
        if (triple) i += 2;
        quote = null;
      }
    } else if (character === "#") comment = true;
    else if (character === "'" || character === '"') {
      quote = character;
      triple = source.startsWith(character.repeat(3), i);
      if (triple) i += 2;
    }
  }
  return !quote && !comment;
}

type Method = {
  label: string;
  insert: (shape: number[] | null) => string;
  result: (shape: number[]) => number[] | null;
  note: string;
  /** Minimum rank for the suggestion to make sense. */
  rank?: number;
};

const same = (shape: number[]) => shape;
const dropLast = (shape: number[]) => shape.slice(0, -1);
const product = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

const METHODS: Method[] = [
  {
    label: "reshape",
    insert: (shape) => `reshape(-1, ${shape?.at(-1) ?? 1})`,
    result: (shape) => [product(shape.slice(0, -1)), shape.at(-1) ?? 1],
    note: "regroup the same elements",
    rank: 1,
  },
  {
    label: "permute",
    insert: (shape) =>
      `permute(${(shape ?? [0, 0])
        .map((_, i, all) => all.length - 1 - i)
        .join(", ")})`,
    result: (shape) => [...shape].reverse(),
    note: "reorder the axes",
    rank: 2,
  },
  {
    label: "transpose",
    insert: () => "transpose(-2, -1)",
    result: (shape) => [...shape.slice(0, -2), shape.at(-1)!, shape.at(-2)!],
    note: "swap two axes",
    rank: 2,
  },
  {
    label: "flatten",
    insert: (shape) => (shape && shape.length > 2 ? "flatten(1)" : "flatten()"),
    result: (shape) =>
      shape.length > 2 ? [shape[0], product(shape.slice(1))] : [product(shape)],
    note: "merge axes into one",
    rank: 1,
  },
  {
    label: "unsqueeze",
    insert: () => "unsqueeze(0)",
    result: (shape) => [1, ...shape],
    note: "insert an axis of size one",
  },
  {
    label: "squeeze",
    insert: () => "squeeze()",
    result: (shape) => shape.filter((size) => size !== 1),
    note: "drop axes of size one",
  },
  {
    label: "sum",
    insert: () => "sum(dim=-1)",
    result: dropLast,
    note: "add along an axis",
    rank: 1,
  },
  {
    label: "mean",
    insert: () => "mean(dim=-1)",
    result: dropLast,
    note: "average along an axis",
    rank: 1,
  },
  {
    label: "amax",
    insert: () => "amax(dim=-1)",
    result: dropLast,
    note: "largest value along an axis",
    rank: 1,
  },
  {
    label: "argmax",
    insert: () => "argmax(dim=-1)",
    result: dropLast,
    note: "position of the largest value",
    rank: 1,
  },
  {
    label: "softmax",
    insert: () => "softmax(dim=-1)",
    result: same,
    note: "weights that sum to one",
    rank: 1,
  },
  {
    label: "cumsum",
    insert: () => "cumsum(dim=-1)",
    result: same,
    note: "running total along an axis",
    rank: 1,
  },
  {
    label: "sort",
    insert: () => "sort(dim=-1)",
    result: () => null,
    note: "ordered values and their positions",
    rank: 1,
  },
  {
    label: "flip",
    insert: () => "flip(-1)",
    result: same,
    note: "reverse the order along an axis",
    rank: 1,
  },
  {
    label: "repeat",
    insert: (shape) =>
      `repeat(2${", 1".repeat(Math.max(0, (shape?.length ?? 1) - 1))})`,
    result: (shape) => (shape.length ? [shape[0] * 2, ...shape.slice(1)] : [2]),
    note: "copy the values",
  },
  {
    label: "expand",
    insert: (shape) => `expand(2, ${(shape ?? []).map(() => -1).join(", ")})`,
    result: (shape) => [2, ...shape],
    note: "broadcast without copying",
    rank: 1,
  },
  {
    label: "roll",
    insert: () => "roll(1, dims=-1)",
    result: same,
    note: "shift with wraparound",
    rank: 1,
  },
  {
    label: "chunk",
    insert: () => "chunk(2, dim=0)",
    result: () => null,
    note: "split into parts",
    rank: 1,
  },
  {
    label: "clone",
    insert: () => "clone()",
    result: same,
    note: "copy into new storage",
  },
  {
    label: "contiguous",
    insert: () => "contiguous()",
    result: same,
    note: "pack values in memory order",
  },
  {
    label: "masked_fill",
    insert: () => "masked_fill(mask, 0.0)",
    result: same,
    note: "overwrite where a mask is true",
  },
  {
    label: "T",
    insert: () => "T",
    result: (shape) => [...shape].reverse(),
    note: "matrix transpose",
    rank: 2,
  },
  {
    label: "shape",
    insert: () => "shape",
    result: () => null,
    note: "the size of every axis",
  },
];

const FUNCTIONS: Record<
  string,
  [label: string, insert: string, note: string][]
> = {
  torch: [
    ["cat", "cat([$, $], dim=0)", "join along an existing axis"],
    ["stack", "stack([$, $], dim=0)", "join along a new axis"],
    ["softmax", "softmax($, dim=-1)", "weights that sum to one"],
    ["where", "where($ > 0, $, 0.0)", "choose between two sources"],
    ["matmul", "matmul($, $.transpose(-2, -1))", "matrix product"],
    ["einsum", 'einsum("...ij,...kj->...ik", $, $)', "named-index product"],
    ["tril", "tril($)", "keep the lower triangle"],
    ["arange", "arange(6)", "0, 1, 2, …"],
    ["zeros", "zeros(2, 3)", "a tensor of zeros"],
    ["ones", "ones(2, 3)", "a tensor of ones"],
    ["randn", "randn(2, 3)", "random normal values"],
    ["tensor", "tensor([1.0, 2.0, 3.0])", "a tensor from a list"],
  ],
  F: [
    ["relu", "relu($)", "zero the negative values"],
    ["gelu", "gelu($)", "smooth activation"],
    ["softmax", "softmax($, dim=-1)", "weights that sum to one"],
    ["linear", "linear($, weight)", "project the last axis"],
    ["conv2d", "conv2d($, kernel, padding=1)", "slide a kernel"],
    ["max_pool2d", "max_pool2d($, 2)", "largest value per window"],
    ["embedding", "embedding(ids, $)", "look up rows by index"],
    ["one_hot", "one_hot(labels, 4)", "class numbers to indicator vectors"],
    ["log_softmax", "log_softmax($, dim=-1)", "log-probabilities"],
    ["cross_entropy", "cross_entropy($, labels)", "score the true class"],
    ["layer_norm", "layer_norm($, $.shape[-1:])", "normalize each group"],
  ],
};

const text = (shape: number[]) => `[${shape.join(", ")}]`;

/**
 * Suggestions for `name.` at the caret. For a tensor whose shape is known from
 * the displayed run, each suggestion shows the shape it would produce.
 */
export function completionsAt(
  source: string,
  caret: number,
  shapes: Record<string, number[]>,
  latest: string,
  limit = 8,
): (CompletionRange & { items: Completion[] }) | null {
  if (!isCodeAt(source, caret)) return null;
  const match = /([A-Za-z_]\w*)\.([A-Za-z_]\w*)?$/.exec(source.slice(0, caret));
  if (!match) return null;
  const [, owner, typed = ""] = match;
  const prefix = typed.toLowerCase();
  const from = caret - typed.length;
  const to = caret + (/^\w*/.exec(source.slice(caret))?.[0].length ?? 0);
  if (Object.prototype.hasOwnProperty.call(FUNCTIONS, owner)) {
    const items = FUNCTIONS[owner]
      .filter(([label]) => label.toLowerCase().startsWith(prefix))
      .map(([label, insert, detail]) => ({
        label,
        insert: insert.replaceAll("$", latest),
        detail,
      }));
    return items.length ? { from, to, items: items.slice(0, limit) } : null;
  }
  if (!Object.prototype.hasOwnProperty.call(shapes, owner)) return null;
  const shape = shapes[owner];
  const items = METHODS.filter(
    (method) =>
      method.label.toLowerCase().startsWith(prefix) &&
      shape.length >= (method.rank ?? 0),
  ).map((method) => {
    const result = method.result(shape);
    return {
      label: method.label,
      insert: method.insert(shape),
      detail: result ? `${text(shape)} → ${text(result)}` : method.note,
    };
  });
  // A finished word needs no suggestion.
  if (items.length === 1 && items[0].insert === typed) return null;
  return items.length ? { from, to, items: items.slice(0, limit) } : null;
}

/** The identifier under a column of a line, if any. */
export function wordAt(line: string, column: number): string | null {
  for (const match of line.matchAll(/[A-Za-z_]\w*/g))
    if (column >= match.index && column < match.index + match[0].length)
      return line[match.index - 1] === "." ? null : match[0];
  return null;
}

/** What a line recorded, as much as contract suggestions need. */
export type RecordedTensor = {
  shape: number[];
  dtype: string;
  numel: number;
  values: (number | string)[];
  minimum?: number | null;
  maximum?: number | null;
  value_source?: string;
  histogram?: {
    counts: number[];
    zeros: number;
    non_finite: number;
    std?: number | null;
  } | null;
};

/** Two significant digits, rounded away from the recorded value. */
function loose(value: number, up: boolean) {
  if (value === 0 || !Number.isFinite(value)) return value;
  const scale = 10 ** (Math.floor(Math.log10(Math.abs(value))) - 1);
  return Number(
    ((up ? Math.ceil : Math.floor)(value / scale) * scale).toPrecision(2),
  );
}

/** Whether every sum over the last axis is 1, as a softmax's are. */
function rowsSumToOne(tensor: RecordedTensor) {
  if (tensor.values.length !== tensor.numel || !tensor.shape.length)
    return false;
  const size = tensor.shape.at(-1)!;
  if (size < 2) return false;
  for (let start = 0; start < tensor.numel; start += size) {
    let sum = 0;
    for (let i = start; i < start + size; i++) sum += Number(tensor.values[i]);
    if (!(Math.abs(sum - 1) <= 1e-4)) return false;
  }
  return true;
}

/**
 * Contract clauses for a line's recorded tensor, offered when a comment is
 * started after it: `#` or `# ra`. Each states what the run recorded, so
 * accepting one pins down today's behaviour for later runs.
 */
export function contractCompletionsAt(
  source: string,
  caret: number,
  tensorAt: (line: number) => RecordedTensor | null | undefined,
  limit = 8,
): (CompletionRange & { items: Completion[] }) | null {
  const before = source.slice(0, caret);
  const lineStart = before.lastIndexOf("\n") + 1;
  const lineText = before.slice(lineStart);
  // Code first, then the comment being started, or a clause after earlier
  // ones: `x = f(y)  # ra` or `x = f(y)  # shape: B, T; ra`.
  const match = /^[^#]*\S[^#]*#(?:[^#]*;)?\s*([a-z(]*)$/.exec(lineText);
  if (!match || !isCodeAt(source, lineStart + lineText.indexOf("#")))
    return null;
  const tensor = tensorAt(before.split("\n").length);
  if (!tensor || tensor.value_source === "shape") return null;
  const typed = match[1];
  const from = caret - typed.length;
  const to = caret + (/^[\w(]*/.exec(source.slice(caret))?.[0].length ?? 0);
  const items: Completion[] = [];
  const shape = tensor.shape.join(", ") || "()";
  items.push({
    label: "shape",
    insert: `shape: ${shape}`,
    detail: "recorded shape",
  });
  const { minimum, maximum, histogram } = tensor;
  if (typeof minimum === "number" && typeof maximum === "number")
    items.push({
      label: "range",
      insert: `range: ${loose(minimum, false)}..${loose(maximum, true)}`,
      detail: `recorded ${minimum.toPrecision(3)} … ${maximum.toPrecision(3)}`,
    });
  if (histogram && !histogram.non_finite)
    items.push({
      label: "finite",
      insert: "finite",
      detail: "no NaN or ∞ recorded",
    });
  if (rowsSumToOne(tensor))
    items.push({
      label: "sums",
      insert: "sums(-1): 1",
      detail: "every row sums to 1",
    });
  if (histogram) {
    const total =
      histogram.counts.reduce((sum, count) => sum + count, 0) +
      histogram.non_finite;
    if (total && histogram.zeros) {
      const share = histogram.zeros / total;
      items.push({
        label: "zeros",
        insert: `zeros: ..${Math.min(100, Math.ceil((share * 100 + 0.01) / 5) * 5)}%`,
        detail: `recorded ${Math.round(share * 1000) / 10}% zeros`,
      });
    }
    if (typeof histogram.std === "number" && histogram.std > 0)
      items.push({
        label: "std",
        insert: `std: ${loose(histogram.std / 2, false)}..${loose(histogram.std * 2, true)}`,
        detail: `recorded σ ${histogram.std.toPrecision(3)}, within ×2`,
      });
  }
  items.push({
    label: "dtype",
    insert: `dtype: ${tensor.dtype.replace(/^torch\./, "")}`,
    detail: "recorded dtype",
  });
  const prefix = typed.toLowerCase();
  const shown = items.filter((item) => item.label.startsWith(prefix));
  if (shown.length === 1 && shown[0].insert === typed) return null;
  return shown.length ? { from, to, items: shown.slice(0, limit) } : null;
}

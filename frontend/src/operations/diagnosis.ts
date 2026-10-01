import type { Operation, Tensor } from "../api/client";
import { product } from "../tensors/coordinates";

export type DiagnosedOperand = {
  label: string;
  name: string;
  shape: number[];
  /** Axes that take part in the mismatch. */
  marks: number[];
  /** Shown when the data type is what went wrong. */
  dtype?: string;
};
export type Diagnosis = {
  title: string;
  explanation: string;
  operands: DiagnosedOperand[];
  /** A change that is known to resolve the problem, when one exists. */
  suggestion: string | null;
};

const text = (shape: number[]) => `[${shape.join(", ")}]`;
const operand = (
  label: string,
  tensor: Tensor,
  marks: number[] = [],
): DiagnosedOperand => ({
  label,
  name: tensor.name,
  shape: tensor.shape,
  marks,
});

/** Right-aligned axis pairs that neither match nor broadcast. */
export function broadcastConflicts(a: number[], b: number[]) {
  const conflicts: { a: number; b: number }[] = [];
  for (let i = 1; i <= Math.min(a.length, b.length); i++) {
    const x = a[a.length - i],
      y = b[b.length - i];
    if (x !== y && x !== 1 && y !== 1)
      conflicts.push({ a: a.length - i, b: b.length - i });
  }
  return conflicts;
}

function broadcast(kind: string, a: Tensor, b: Tensor): Diagnosis | null {
  const conflicts = broadcastConflicts(a.shape, b.shape);
  if (!conflicts.length) return null;
  const [short, long] = a.shape.length <= b.shape.length ? [a, b] : [b, a];
  let suggestion: string | null = null;
  // Trailing unit axes often turn a per-row vector into a broadcastable column.
  for (
    let extra = 1;
    extra <= long.shape.length - short.shape.length;
    extra++
  ) {
    const padded = [...short.shape, ...Array(extra).fill(1)];
    if (!broadcastConflicts(padded, long.shape).length) {
      suggestion =
        extra === 1
          ? `${short.name}.unsqueeze(-1) has shape ${text(padded)} and broadcasts with ${text(long.shape)}.`
          : `${short.name}.reshape(${padded.join(", ")}) broadcasts with ${text(long.shape)}.`;
      break;
    }
  }
  const first = conflicts[0];
  return {
    title: "These shapes cannot broadcast",
    explanation: `${kind} lines shapes up from the last axis. Each pair of sizes must be equal, or one of them must be 1. Here ${a.shape[first.a]} meets ${b.shape[first.b]}.`,
    operands: [
      operand(
        "Left",
        a,
        conflicts.map((c) => c.a),
      ),
      operand(
        "Right",
        b,
        conflicts.map((c) => c.b),
      ),
    ],
    suggestion,
  };
}

function matmul(a: Tensor, b: Tensor): Diagnosis | null {
  if (!a.shape.length || !b.shape.length) return null;
  const leftAxis = a.shape.length - 1;
  const rightAxis = b.shape.length > 1 ? b.shape.length - 2 : 0;
  const left = a.shape[leftAxis],
    right = b.shape[rightAxis];
  if (left === right) {
    const batch = broadcastConflicts(
      a.shape.slice(0, -2),
      b.shape.slice(0, -2),
    );
    if (!batch.length) return null;
    return {
      title: "The batch dimensions cannot broadcast",
      explanation:
        "The matrices line up, but the leading batch dimensions must be equal or 1.",
      operands: [
        operand(
          "Left",
          a,
          batch.map((c) => c.a),
        ),
        operand(
          "Right",
          b,
          batch.map((c) => c.b),
        ),
      ],
      suggestion: null,
    };
  }
  const swapped = (t: Tensor) => [
    ...t.shape.slice(0, -2),
    t.shape.at(-1)!,
    t.shape.at(-2)!,
  ];
  let suggestion: string | null = null;
  if (b.shape.length > 1 && b.shape.at(-1) === left)
    suggestion = `${b.name}.transpose(-2, -1) has shape ${text(swapped(b))}, which lines up with ${left}.`;
  else if (a.shape.length > 1 && a.shape.at(-2) === right)
    suggestion = `${a.name}.transpose(-2, -1) has shape ${text(swapped(a))}, which lines up with ${right}.`;
  return {
    title: "The inner dimensions do not match",
    explanation: `A matrix product pairs each row of the left tensor with each column of the right one, so the left's last size (${left}) must equal the right's ${b.shape.length > 1 ? "second-to-last" : "only"} size (${right}).`,
    operands: [
      operand("Left", a, [leftAxis]),
      operand("Right", b, [rightAxis]),
    ],
    suggestion,
  };
}

function regroup(op: Operation, input: Tensor): Diagnosis | null {
  const requested = op.arguments.shape;
  if (!Array.isArray(requested) || requested.some((n) => !Number.isInteger(n)))
    return null;
  const shape = requested as number[];
  const known = product(shape.filter((n) => n !== -1));
  const inferred = shape.includes(-1);
  if (inferred ? known && input.numel % known === 0 : known === input.numel)
    return null;
  return {
    title: "The element count does not fit",
    explanation: inferred
      ? `-1 stands for whatever size is left over, but ${input.numel} elements cannot be divided evenly by ${known}.`
      : `${op.kind} only regroups elements. ${text(input.shape)} holds ${input.numel}, and ${text(shape)} needs ${known}.`,
    operands: [
      operand("Input", input),
      {
        label: "Requested",
        name: "shape",
        shape,
        marks: shape.map((_, i) => i),
      },
    ],
    suggestion: inferred
      ? null
      : `Sizes whose product is ${input.numel} will work, for example ${text(input.numel % 2 ? [-1] : [-1, 2])}.`,
  };
}

function join(op: Operation, tensors: Tensor[]): Diagnosis | null {
  if (tensors.length < 2) return null;
  const first = tensors[0];
  const stacking = op.kind === "stack";
  const rank = first.shape.length;
  const raw = Number(op.arguments.dim ?? 0);
  const axis = stacking ? -1 : ((raw % rank) + rank) % rank;
  const operands = tensors
    .slice(0, 6)
    .map((tensor, i) =>
      operand(
        `Tensor ${i}`,
        tensor,
        tensor.shape.length !== rank
          ? tensor.shape.map((_, j) => j)
          : tensor.shape.flatMap((size, j) =>
              j !== axis && size !== first.shape[j] ? [j] : [],
            ),
      ),
    );
  if (!operands.some((item) => item.marks.length)) return null;
  // Mark the reference sizes too, so both sides of the mismatch stand out.
  const disputed = new Set(operands.flatMap((item) => item.marks));
  operands[0].marks = [...disputed].filter((j) => j < rank);
  return {
    title: stacking
      ? "Stacked tensors need identical shapes"
      : "Only the joined axis may differ",
    explanation: stacking
      ? "stack places the tensors side by side along a new axis, so every tensor must have the same shape."
      : `cat extends axis ${axis}. Every other axis must have the same size in all tensors.`,
    operands,
    suggestion: null,
  };
}

function convolution(op: Operation, input: Tensor, weight: Tensor) {
  const spatial = op.kind === "conv1d" ? 1 : 2;
  if (weight.shape.length !== spatial + 2) return null;
  const rank = input.shape.length;
  if (rank !== spatial + 1 && rank !== spatial + 2)
    return {
      title: "The input has the wrong number of dimensions",
      explanation: `${op.kind} reads [batch, channels, ${spatial === 1 ? "length" : "height, width"}], or the same without batch. This input has ${rank} dimensions.`,
      operands: [operand("Input", input), operand("Kernel", weight)],
      suggestion: null,
    } satisfies Diagnosis;
  const channelAxis = rank - spatial - 1;
  const groups = Number(op.arguments.groups ?? 1) || 1;
  if (input.shape[channelAxis] === weight.shape[1] * groups) return null;
  return {
    title: "The channel counts do not match",
    explanation: `The kernel expects ${weight.shape[1] * groups} input channels, but axis ${channelAxis} of the input has ${input.shape[channelAxis]}.${rank === spatial + 1 ? " Without a batch axis, the first axis is read as channels." : ""}`,
    operands: [
      operand("Input", input, [channelAxis]),
      operand("Kernel", weight, [1]),
    ],
    // Adding a batch axis leaves the channel count unchanged, so it cannot
    // repair this mismatch. Keep the recorded channel guidance instead.
    suggestion: null,
  } satisfies Diagnosis;
}

const ELEMENTWISE = new Set([
  "add",
  "sub",
  "mul",
  "div",
  "true_divide",
  "pow",
  "maximum",
  "minimum",
  "eq",
  "ne",
  "gt",
  "ge",
  "lt",
  "le",
  "where",
  "masked_fill",
]);

/** Explain a failed operation from its recorded operand shapes. */
const typed = (label: string, tensor: Tensor): DiagnosedOperand => ({
  ...operand(label, tensor),
  dtype: tensor.dtype,
});
const isFloat = (tensor: Tensor) => tensor.dtype.startsWith("float");

/** Failures that are about data types, memory order, or out-of-range positions. */
function beyondShapes(op: Operation, inputs: Tensor[]): Diagnosis | null {
  const message = op.error ?? "";
  const [a, b] = inputs;
  if (
    op.kind === "view" &&
    /not compatible with input tensor's size and stride/.test(message)
  ) {
    const requested = Array.isArray(op.arguments.shape)
      ? (op.arguments.shape as number[]).join(", ")
      : "…";
    return {
      title: "The values are no longer stored in reading order",
      explanation: `view only relabels memory that is already in order. After a permute or transpose, ${a.name} reads its elements in a different order than they are stored, so its ${a.numel} values cannot simply be regrouped.`,
      operands: [operand("Input", a)],
      suggestion: `${a.name}.reshape(${requested}) copies when it has to. ${a.name}.contiguous().view(${requested}) does the same in two steps.`,
    };
  }
  if (
    /scalar types: Long, Int|only applicable to index tensor|must be long, int, byte or bool|expected target dtype to be Long/.test(
      message,
    )
  ) {
    // The index is the first operand, except for subscripts and loss targets.
    const index =
      ["__getitem__", "cross_entropy", "nll_loss"].includes(op.kind) && b
        ? b
        : a;
    return {
      title: "Positions must be whole numbers",
      explanation: `${index.name} holds ${index.dtype} values. A position, row number, or class number has to be an integer tensor (int64).`,
      operands: inputs
        .slice(0, 3)
        .map((tensor, i) => typed(i ? `Operand ${i}` : "Input", tensor)),
      suggestion: `${index.name}.long() converts it, dropping any fraction.`,
    };
  }
  if (/expected condition to be a boolean/.test(message))
    return {
      title: "A condition must be true or false",
      explanation: `${a.name} holds ${a.dtype} values. where chooses by a Boolean tensor.`,
      operands: [typed("Condition", a)],
      suggestion: `A comparison such as ${a.name} > 0 produces one.`,
    };
  if (
    b &&
    a.dtype !== b.dtype &&
    /expected scalar type|same dtype|expected m1 and m2/.test(message)
  ) {
    // Convert toward the more general type: integers to floats, narrow to wide.
    const [from, to] =
      isFloat(a) === isFloat(b)
        ? a.dtype < b.dtype
          ? [a, b]
          : [b, a]
        : isFloat(a)
          ? [b, a]
          : [a, b];
    return {
      title: "The operands have different data types",
      explanation: `${a.name} is ${a.dtype} and ${b.name} is ${b.dtype}. This operation does not mix them automatically.`,
      operands: [typed("Left", a), typed("Right", b)],
      suggestion: `${from.name}.to(${to.name}.dtype) converts ${from.name} to ${to.dtype}.`,
    };
  }
  if (/must be either a floating point or complex dtype/.test(message))
    return {
      title: "This needs fractional values",
      explanation: `${a.name} holds ${a.dtype} values, and the result of ${op.kind} is generally not a whole number.`,
      operands: [typed("Input", a)],
      suggestion: `${a.name}.float().${op.kind}(…) converts it first.`,
    };
  const bounds =
    /index (-?\d+) is out of bounds for dimension (\d+) with size (\d+)/.exec(
      message,
    );
  if (bounds) {
    const [, index, axis, size] = bounds.map(Number);
    return {
      title: "That position does not exist",
      explanation: `Axis ${axis} has ${size} ${size === 1 ? "position" : "positions"}, numbered 0 to ${size - 1} (or −${size} to −1 from the end). Position ${index} is outside it.`,
      operands: [operand("Input", a, axis < a.shape.length ? [axis] : [])],
      suggestion: null,
    };
  }
  if (op.kind === "embedding" && b && /index out of range/.test(message)) {
    const ids = a.values.filter((v): v is number => typeof v === "number");
    const known = ids.length === a.numel && ids.length > 0;
    const worst = known ? ids.find((v) => v < 0 || v >= b.shape[0]) : undefined;
    return {
      title: "An index points past the table",
      explanation: `The table has ${b.shape[0]} rows, numbered 0 to ${b.shape[0] - 1}.${worst !== undefined ? ` ${a.name} contains ${worst}.` : ""}`,
      operands: [operand("Indices", a), operand("Table", b, [0])],
      suggestion: null,
    };
  }
  const target = /Target (-?\d+) is out of bounds/.exec(message);
  if (target) {
    const classes = a.shape.at(-1) ?? 0;
    return {
      title: "A class number is out of range",
      explanation: `The scores have ${classes} classes, numbered 0 to ${classes - 1}. A target is ${target[1]}.`,
      operands: [
        operand("Scores", a, [a.shape.length - 1]),
        ...(b ? [operand("Targets", b)] : []),
      ],
      suggestion: null,
    };
  }
  const expanded =
    /expanded size of the tensor \((\d+)\) must match the existing size \((\d+)\) at non-singleton dimension (\d+)\.\s+Target sizes: \[([\d, ]+)\]/.exec(
      message,
    );
  if (expanded) {
    const [, wanted, existing, axis] = expanded.map(Number);
    const sizes = expanded[4].split(",").map(Number);
    const offset = sizes.length - a.shape.length;
    const repeats = sizes.map((size, i) =>
      i < offset ? size : size / a.shape[i - offset],
    );
    return {
      title: "Only axes of size one can stretch",
      explanation: `expand reuses one stored value along an axis, so that axis must have size 1. Here an axis has size ${existing} and cannot become ${wanted}.`,
      operands: [
        operand("Input", a, axis - offset >= 0 ? [axis - offset] : []),
        {
          label: "Requested",
          name: "size",
          shape: sizes,
          marks: [axis],
        },
      ],
      suggestion: repeats.every((r) => Number.isInteger(r) && r > 0)
        ? `${a.name}.repeat(${repeats.join(", ")}) copies the values to reach ${text(sizes)}.`
        : null,
    };
  }
  const split = /don't multiply up to the size of dim (\d+) \((\d+)\)/.exec(
    message,
  );
  if (split) {
    const axis = Number(split[1]);
    return {
      title: "The new sizes do not fill the axis",
      explanation: `Axis ${axis} has size ${split[2]}; the sizes it is split into must multiply to exactly that.`,
      operands: [operand("Input", a, [axis])],
      suggestion: null,
    };
  }
  return null;
}

export function diagnose(
  op: Operation,
  tensors: Record<string, Tensor>,
): Diagnosis | null {
  if (op.status !== "error") return null;
  const inputs = op.inputs.map((id) => tensors[id]).filter(Boolean);
  if (!inputs.length) return null;
  const [a, b] = inputs;
  let result: Diagnosis | null = null;
  if (["matmul", "mm", "bmm"].includes(op.kind) && b) result = matmul(a, b);
  else if (op.kind === "linear" && b && b.shape.length === 2) {
    if (a.shape.at(-1) !== b.shape[1])
      result = {
        title: "The feature counts do not match",
        explanation: `A linear layer's weight has shape [out_features, in_features]. It expects ${b.shape[1]} input features, but the input's last axis has ${a.shape.at(-1)}.`,
        operands: [
          operand("Input", a, [a.shape.length - 1]),
          operand("Weight", b, [1]),
        ],
        suggestion: null,
      };
  } else if (["reshape", "view"].includes(op.kind)) result = regroup(op, a);
  else if (["cat", "concat", "concatenate", "stack"].includes(op.kind))
    result = join(op, inputs);
  else if (["conv1d", "conv2d"].includes(op.kind) && b)
    result = convolution(op, a, b);
  else if (op.kind === "permute" && Array.isArray(op.arguments.dims)) {
    if (op.arguments.dims.length !== a.shape.length)
      result = {
        title: "permute needs every axis exactly once",
        explanation: `This tensor has ${a.shape.length} axes, but ${op.arguments.dims.length} were listed.`,
        operands: [operand("Input", a)],
        suggestion: `List all of them, for example permute(${a.shape
          .map((_, i) => i)
          .reverse()
          .join(", ")}).`,
      };
  } else if (ELEMENTWISE.has(op.kind) && b) result = broadcast(op.kind, a, b);
  result ??= beyondShapes(op, inputs);
  if (result) return result;
  const range = /Dimension out of range/.test(op.error ?? "");
  return {
    title: range
      ? "That axis does not exist"
      : "This operation stopped the run",
    explanation: range
      ? `This tensor has ${a.shape.length} axes, numbered 0 to ${a.shape.length - 1} (or -${a.shape.length} to -1 from the end).`
      : (op.error ?? "PyTorch raised an error."),
    operands: inputs
      .slice(0, 6)
      .map((tensor, i) => operand(`Operand ${i}`, tensor)),
    suggestion: null,
  };
}

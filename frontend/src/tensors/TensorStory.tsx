import { useContext, useMemo } from "react";
import type { Run, Tensor } from "../api/client";
import { bytesText } from "../journey/cost";
import { tensorInsights } from "../shell/insights";
import type { Problem } from "../shell/problems";
import { describeAxis, type AxisStory } from "./axisLineage";
import { formatValue } from "./coordinates";
import { SymbolicContext } from "./InkShape";
import { LineageContext } from "./LineageContext";
import { tensorBytes } from "./memory";
import { TensorUseContext, tensorUses } from "./TensorUseContext";
import { TokenContext, type TokenAxes } from "./TokenContext";

type Trace = Run["trace"];

const shapeText = (tensor: Tensor) => `[${tensor.shape.join(", ")}]`;
const list = (items: string[]) =>
  items.length <= 1
    ? (items[0] ?? "")
    : `${items.slice(0, -1).join(", ")} and ${items.at(-1)}`;

/**
 * A tensor in a few plain sentences: where it comes from, what its axes are,
 * what its values look like, what it costs, who reads it, and what the run
 * notes say about its step. Each sentence comes from something the run
 * recorded; nothing is guessed.
 */
export function explainTensor(
  trace: Trace,
  tensor: Tensor,
  context: {
    lineage?: ((tensorId: string) => AxisStory[]) | null;
    symbolic?: (string | null)[] | null;
    tokens?: TokenAxes | null;
    notes?: Problem[];
  } = {},
): string[] {
  const told: string[] = [];
  const usesOf = tensorUses(trace);
  const uses = usesOf(tensor.id);
  const op = uses.made
    ? trace.operations.find((each) => each.id === uses.made!.id)
    : undefined;
  const named = (id: string) => {
    const other = trace.tensors[id];
    return other ? `${other.name} ${shapeText(other)}` : null;
  };

  // Where it comes from.
  if (trace.input_ids.includes(tensor.id))
    told.push(
      `${tensor.name} is an input of the model, ${shapeText(tensor)} ${tensor.dtype}.`,
    );
  else if (tensor.role === "parameter")
    told.push(
      `${tensor.name} is a weight of the model, ${shapeText(tensor)} ${tensor.dtype}.`,
    );
  else if (op) {
    const from = op.inputs.flatMap((id) => named(id) ?? []);
    told.push(
      `${tensor.name} is made at step ${op.index + 1} by ${op.kind}${op.source?.line ? ` (line ${op.source.line})` : ""}${from.length ? ` from ${list(from)}` : ""}.`,
    );
  }

  // What its axes are.
  const stories = context.lineage?.(tensor.id) ?? [];
  const axes = tensor.shape.map((size, axis) => {
    const own = tensor.axes[axis];
    const story = stories[axis];
    // Only a traced origin is worth citing, not a bare note ("broadcast").
    const origin = story?.terms.length ? describeAxis(story) : null;
    const name = own && !/^axis \d+$/.test(own) ? own : null;
    const words = context.tokens?.(tensor, axis);
    const tail = words
      ? words.vocabulary
        ? ", one per word of the vocabulary"
        : `, one per word: ${words.words.slice(0, 6).join(" ")}${words.words.length > 6 ? " …" : ""}`
      : "";
    const from =
      origin && origin !== "unknown" && origin !== name
        ? ` from ${origin}`
        : "";
    return `${name ?? `axis ${axis}`} (${size}${from}${tail})`;
  });
  if (axes.length)
    told.push(
      `Its ${axes.length === 1 ? "axis is" : "axes are"} ${list(axes)}.`,
    );
  if (
    context.symbolic &&
    context.symbolic.some((term) => term && /[A-Z]/.test(term))
  )
    told.push(
      `In the input's sizes it is [${context.symbolic.map((term, axis) => term ?? tensor.shape[axis]).join(", ")}].`,
    );

  // What its values look like.
  const histogram = tensor.histogram;
  const low = tensor.minimum ?? histogram?.low;
  const high = tensor.maximum ?? histogram?.high;
  if (
    tensor.value_source !== "shape" &&
    typeof low === "number" &&
    typeof high === "number"
  ) {
    const total = histogram
      ? histogram.counts.reduce((sum, count) => sum + count, 0) +
        histogram.non_finite
      : 0;
    const zeros = histogram && total ? histogram.zeros / total : 0;
    const parts = [`from ${formatValue(low)} to ${formatValue(high)}`];
    if (typeof histogram?.mean === "number")
      parts.push(`mean ${formatValue(histogram.mean)}`);
    if (typeof histogram?.std === "number")
      parts.push(`σ ${formatValue(histogram.std)}`);
    told.push(
      `Its values run ${parts.join(", ")}${zeros >= 0.01 ? `; ${Math.round(zeros * 100)}% are zero` : ""}${histogram?.non_finite ? `; ${histogram.non_finite} are NaN or infinite` : ""}.`,
    );
  } else if (tensor.value_source === "shape")
    told.push("This run recorded its shape only, not its values.");

  // What it costs.
  const bytes = tensorBytes(tensor);
  const sharer = Object.values(trace.tensors).find(
    (other) =>
      other.id !== tensor.id &&
      !!tensor.storage_id &&
      other.storage_id === tensor.storage_id &&
      (trace.input_ids.includes(other.id) ||
        other.role === "parameter" ||
        (usesOf(other.id).made?.step ?? Infinity) <
          (uses.made?.step ?? Infinity)),
  );
  if (bytes !== null)
    told.push(
      sharer
        ? `It is a view of ${sharer.name}, so its ${bytesText(bytes)} are not new memory.`
        : `It takes ${bytesText(bytes)} of memory.`,
    );

  // Who reads it.
  const readers = uses.read.map((step) => {
    const reader = trace.operations.find((each) => each.id === step.id);
    const made = reader?.outputs[0]
      ? trace.tensors[reader.outputs[0]]?.name
      : null;
    return `step ${step.step} (${step.kind}${made ? ` → ${made}` : ""})`;
  });
  const output = trace.output_ids.includes(tensor.id);
  if (readers.length)
    told.push(
      `${readers.length === 1 ? "It is read by" : `${readers.length} steps read it:`} ${list(readers.slice(0, 4))}${readers.length > 4 ? ", and more" : ""}${output ? ", and it is one of the model's results" : ""}.`,
    );
  else if (output) told.push("It is one of the model's results.");
  else if (op)
    told.push("Nothing reads it afterwards, and the model does not return it.");

  // What the run notes say about its step.
  const notes = (context.notes ?? []).filter(
    (note) => op && note.node === op.id,
  );
  if (notes.length)
    told.push(
      `Run notes on its step: ${list(notes.map((note) => note.title))}.`,
    );
  return told;
}

/** Run notes, worked out once per trace. */
const noted = new WeakMap<Trace, Problem[]>();

/** "About" a tensor, in the card's details: {@link explainTensor}. */
export function TensorStory({ tensor }: { tensor: Tensor }) {
  const flow = useContext(TensorUseContext);
  const lineage = useContext(LineageContext);
  const symbolic = useContext(SymbolicContext);
  const tokens = useContext(TokenContext);
  const trace = flow?.trace;
  const sentences = useMemo(() => {
    if (!trace || !trace.tensors[tensor.id]) return [];
    let notes = noted.get(trace);
    if (!notes) {
      notes = tensorInsights({ project: {}, trace } as unknown as Run);
      noted.set(trace, notes);
    }
    return explainTensor(trace, tensor, {
      lineage,
      symbolic: symbolic?.(tensor.id) ?? null,
      tokens,
      notes,
    });
  }, [trace, tensor, lineage, symbolic, tokens]);
  if (!sentences.length) return null;
  return (
    <div className="tensor-story" aria-label={`About ${tensor.name}`}>
      {sentences.map((sentence, at) => (
        <p key={at}>{sentence}</p>
      ))}
    </div>
  );
}

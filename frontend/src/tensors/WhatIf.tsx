import { createContext, useContext, useEffect, useState } from "react";
import type {
  InputEdit,
  Knockout,
  KnockoutSweep,
  LearnStep,
  SweepResult,
  Tensor,
} from "../api/client";
import { formatValue, unravel } from "./coordinates";
import { asNumber } from "./margins";
import { TensorUseContext } from "./TensorUseContext";
import { usePredictionAxes } from "./TokenContext";

/** What the input's tensor card needs to try another value in a cell. */
export type WhatIfControl = {
  /** The model input's tensor in the displayed run. */
  inputId: string;
  /** A sentence input's words, by token id; null for numbers. */
  vocabulary: string[] | null;
  /** Cells the displayed what-if run set, if it is one. */
  edits: InputEdit[];
  /** The recorded value of a cell, before any what-if. */
  recorded: (index: number) => number | undefined;
  busy: boolean;
  /** Runs the code again with this cell set, keeping earlier what-if cells. */
  run: (edit: InputEdit) => void;
  /** Runs the code again after training steps on every weight. */
  learn: (step: LearnStep) => void;
  /** The training the displayed what-if run took, and the value's curve. */
  learned: { step: LearnStep; curve: (number | null)[] } | null;
  /** Runs the code again with one step's result replaced; null undoes it. */
  knockout: (knockout: Knockout | null) => void;
  /** The knockout the displayed what-if run took. */
  knocked: Knockout | null;
  /** The recorded run what-ifs vary, and sweeps measure against. */
  baseRunId: string;
  /** The run a patch comes from: the one recorded before. */
  patchFrom: string | null;
  /** Each slice of a step's result knocked out in turn; null when busy. */
  sweep: (sweep: KnockoutSweep) => Promise<SweepResult | null>;
  /** Back to the recorded run. */
  leave: (() => void) | null;
};

export const WhatIfContext = createContext<WhatIfControl | null>(null);

/**
 * Under the input's selected cell: another value for it, run through the same
 * code without saving, so every result shows what that one value changes.
 */
export function WhatIf({
  tensor,
  index,
  value,
}: {
  tensor: Tensor;
  index: number;
  value: number | string | boolean | undefined;
}) {
  const control = useContext(WhatIfContext);
  const current = asNumber(value);
  const [text, setText] = useState("");
  // A new cell starts from its own value.
  useEffect(
    () => setText(Number.isFinite(current) ? formatValue(current) : ""),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [tensor.id, index],
  );
  if (!control || control.inputId !== tensor.id || value === undefined)
    return null;
  const coords = `${tensor.name}[${unravel(index, tensor.shape).join(", ")}]`;
  const recorded = control.recorded(index);
  const edited = control.edits.find((edit) => edit.index === index);
  const vocabulary = control.vocabulary;
  const word = (id: number | undefined) =>
    vocabulary && id !== undefined ? (vocabulary[id] ?? `${id}`) : null;
  const parsed = Number(text);
  const valid =
    text.trim() !== "" &&
    Number.isFinite(parsed) &&
    (tensor.dtype !== "int64" || Number.isInteger(parsed));
  const submit = (next: number) => {
    if (Number.isFinite(next) && next !== current)
      control.run({ index, value: next });
  };
  return (
    <form
      className="what-if"
      onSubmit={(event) => {
        event.preventDefault();
        if (valid && text !== formatValue(current)) submit(parsed);
      }}
    >
      <label className="what-if-label">
        What if {coords} =
        {vocabulary ? (
          <select
            value={Number.isFinite(current) ? current : ""}
            disabled={control.busy}
            onChange={(event) => submit(Number(event.target.value))}
          >
            {vocabulary.map((each, id) => (
              <option key={id} value={id}>
                {each}
              </option>
            ))}
          </select>
        ) : (
          <input
            value={text}
            inputMode="decimal"
            spellCheck={false}
            disabled={control.busy}
            aria-invalid={text.trim() !== "" && !valid}
            onChange={(event) => setText(event.target.value)}
          />
        )}
      </label>
      {!vocabulary && (
        <button
          type="submit"
          disabled={
            control.busy ||
            !valid ||
            parsed === current ||
            text === formatValue(current)
          }
          title="Run the same code with this cell set; nothing is saved"
        >
          Try
        </button>
      )}
      {edited && recorded !== undefined && (
        <span className="what-if-was">
          recorded {word(recorded) ?? formatValue(recorded)}
        </span>
      )}
    </form>
  );
}

/** Whether training steps can aim at this tensor's values. */
export function canTrain(
  control: WhatIfControl | null,
  tensor: Tensor,
): control is WhatIfControl {
  return (
    !!control &&
    tensor.dtype.startsWith("float") &&
    tensor.role !== "parameter" &&
    tensor.id !== control.inputId
  );
}

/**
 * Under a result's selected cell: one step of gradient descent on every
 * weight toward raising (or lowering) that value, run as a what-if, so Diff
 * and the change lens show what one training step changes everywhere.
 */
export function LearnControls({
  tensor,
  index,
}: {
  tensor: Tensor;
  index: number;
}) {
  const control = useContext(WhatIfContext);
  const flow = useContext(TensorUseContext);
  const predictions = usePredictionAxes(tensor);
  const [rate, setRate] = useState("0.01");
  const [steps, setSteps] = useState("1");
  // A probability trains on its logarithm: one step of cross-entropy.
  const probability = /^(softmax|sigmoid)$/.test(
    flow?.uses(tensor.id).made?.kind ?? "",
  );
  if (!canTrain(control, tensor)) return null;
  const parsed = Number(rate);
  const count = Number(steps);
  const valid =
    Number.isFinite(parsed) &&
    parsed > 0 &&
    parsed <= 100 &&
    Number.isInteger(count) &&
    count >= 1 &&
    count <= 100;
  const learned =
    control.learned?.step.tensor_id === tensor.id &&
    (control.learned.step.sentence || control.learned.step.index === index)
      ? control.learned
      : null;
  const coords = `${tensor.name}[${unravel(index, tensor.shape).join(", ")}]`;
  const step = (direction: 1 | -1) =>
    valid &&
    control.learn({
      tensor_id: tensor.id,
      index,
      rate: parsed,
      direction,
      log: probability,
      steps: count,
      sentence: false,
    });
  // Scores over a vocabulary at each word: train every word on the next one.
  const sentence = () =>
    valid &&
    control.learn({
      tensor_id: tensor.id,
      index: 0,
      rate: parsed,
      direction: 1,
      log: probability,
      steps: count,
      sentence: true,
    });
  return (
    <div
      className="what-if learn-step"
      title="Steps of gradient descent on every weight, then the same run again; nothing is saved"
    >
      <span className="what-if-label">
        Train {count === 1 ? "one step" : `${count} steps`}{" "}
        {probability ? "to make" : "on"} {coords}
        {probability ? "" : ":"}
      </span>
      <button
        type="button"
        disabled={control.busy || !valid}
        onClick={() => step(1)}
        title={
          probability
            ? "Raise log p: one step of cross-entropy toward this outcome"
            : "Move every weight along ∂ value / ∂ weight"
        }
      >
        {probability ? "↑ likelier" : "↑ raise"}
      </button>
      <button
        type="button"
        disabled={control.busy || !valid}
        onClick={() => step(-1)}
      >
        {probability ? "↓ less likely" : "↓ lower"}
      </button>
      <label className="what-if-label">
        rate
        <input
          value={rate}
          inputMode="decimal"
          spellCheck={false}
          aria-invalid={!valid}
          onChange={(event) => setRate(event.target.value)}
          style={{ width: 48 }}
        />
      </label>
      <label className="what-if-label">
        steps
        <input
          value={steps}
          inputMode="numeric"
          spellCheck={false}
          aria-invalid={
            !(Number.isInteger(count) && count >= 1 && count <= 100)
          }
          onChange={(event) => setSteps(event.target.value)}
          style={{ width: 36 }}
        />
      </label>
      {predictions && (
        <button
          type="button"
          disabled={control.busy || !valid}
          onClick={sentence}
          title="Every word learns to predict the word after it (mean cross-entropy); the curve is the loss"
        >
          ↻ train on the sentence
        </button>
      )}
      {learned && learned.curve.length > 1 && (
        <LearnCurve
          curve={learned.curve}
          // A loss should fall.
          direction={learned.step.sentence ? -1 : learned.step.direction}
          label={learned.step.sentence ? "loss" : undefined}
        />
      )}
    </div>
  );
}

/**
 * The value training aimed at, before each step and after the last: a
 * learning curve, on a log scale when it spans more than a hundredfold.
 */
export function LearnCurve({
  curve,
  direction,
  label,
}: {
  curve: (number | null)[];
  direction: number;
  label?: string;
}) {
  const values = curve.map((value) => value ?? NaN);
  const finite = values.filter(Number.isFinite);
  if (finite.length < 2) return null;
  const positive = finite.every((value) => value > 0);
  const low = Math.min(...finite),
    high = Math.max(...finite);
  const log = positive && high / low > 100;
  const scale = (value: number) => (log ? Math.log10(value) : value);
  const bottom = scale(low),
    top = scale(high);
  const width = 140,
    height = 30;
  const x = (at: number) => 2 + (at / (values.length - 1)) * (width - 4);
  const y = (value: number) =>
    top === bottom
      ? height / 2
      : 2 + (1 - (scale(value) - bottom) / (top - bottom)) * (height - 4);
  const points = values
    .map((value, at) =>
      Number.isFinite(value)
        ? `${x(at).toFixed(1)},${y(value).toFixed(1)}`
        : null,
    )
    .filter(Boolean)
    .join(" ");
  const first = finite[0],
    last = finite.at(-1)!;
  const went = direction > 0 ? last > first : last < first;
  return (
    <span
      className="learn-curve"
      title={`Before each of ${values.length - 1} steps and after the last${log ? " (log scale)" : ""}`}
    >
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <polyline points={points} />
        <circle cx={x(values.length - 1)} cy={y(last)} r={2.5} />
      </svg>
      <span className={went ? "" : "learn-curve-wrong"}>
        {label ? `${label} ` : ""}
        {formatValue(first)} → {formatValue(last)}
      </span>
    </span>
  );
}

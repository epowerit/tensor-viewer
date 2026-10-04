import { useContext, useEffect, useState } from "react";
import type { Knockout, Run, SweepResult, Tensor } from "../api/client";
import { axisName } from "./knockoutText";
import { formatValue } from "./coordinates";
import { TensorUseContext } from "./TensorUseContext";
import { TokenContext } from "./TokenContext";
import { WhatIfContext } from "./WhatIf";

type Mode = Knockout["mode"];

/** Sweeps already asked for, by run, step, result, axis, and mode. */
const sweeps = new Map<string, Promise<SweepResult | null>>();
/**
 * The slice, mode, and sweep last chosen at each step's result, so they stay
 * while a knockout's what-if run takes the recorded run's place.
 */
const chosen = new Map<
  string,
  { axis: number | null; mode: Mode; sweep: string | null }
>();

const percent = (share: number) =>
  share >= 0.1
    ? `${Math.round(share * 100)}%`
    : share >= 0.001
      ? `${(share * 100).toPrecision(2)}%`
      : share > 0
        ? `${(share * 100).toExponential(0)}%`
        : "0";

/** The step that made this result, and which of its results it is. */
export function knockedStep(
  flow: {
    uses: (id: string) => { made: { id: string } | null };
    trace?: Run["trace"];
  } | null,
  tensor: Tensor,
) {
  const made = flow?.uses(tensor.id).made;
  const op = made
    ? flow?.trace?.operations.find((each) => each.id === made.id)
    : undefined;
  const output = op ? op.outputs.indexOf(tensor.id) : -1;
  return op && output >= 0 && tensor.role !== "parameter"
    ? { op, output }
    : null;
}

/**
 * Under a step's result: what if the run had put something else there as it
 * was made, before any later step read it. Zero it, put in its mean, or patch
 * in the same step's result from the run before, in the whole of it or in
 * the slice through the selected cell along one axis (one head, one word).
 * A sweep knocks out every slice along that axis in turn and shows how far
 * each moved the model's output: which heads or words the result leans on.
 */
export function KnockoutControls({
  tensor,
  coords,
}: {
  tensor: Tensor;
  coords: number[];
}) {
  const control = useContext(WhatIfContext);
  const flow = useContext(TensorUseContext);
  const tokens = useContext(TokenContext);
  const made = flow?.uses(tensor.id).made;
  const place = `${made?.id}|${tensor.name}`;
  const remembered = chosen.get(place);
  const [axis, chooseAxis] = useState<number | null>(remembered?.axis ?? null);
  const [mode, chooseMode] = useState<Mode>(remembered?.mode ?? "zero");
  const [swept, setSwept] = useState<{
    key: string;
    result?: SweepResult | null;
  } | null>(null);
  const remember = (next: Partial<NonNullable<typeof remembered>>) =>
    chosen.set(place, {
      axis,
      mode,
      sweep: chosen.get(place)?.sweep ?? null,
      ...next,
    });
  const setAxis = (next: number | null) => {
    chooseAxis(next);
    remember({ axis: next });
  };
  const setMode = (next: Mode) => {
    chooseMode(next);
    remember({ mode: next });
  };
  const found = knockedStep(flow, tensor);
  const op = found?.op;
  const output = found?.output ?? -1;
  // Sweeps measure against the recorded run, even while a what-if shows.
  const runId = control?.baseRunId ?? "";
  const key = op
    ? `${runId}|${op.index}|${output}|${axis}|${mode}|${control?.patchFrom ?? ""}`
    : "";
  // A slice past this tensor's axes (another tensor's choice) is no slice.
  useEffect(() => {
    if (axis !== null && axis >= tensor.shape.length) setAxis(null);
  }, [axis, tensor.shape.length]);
  useEffect(() => {
    if (mode === "patch" && !control?.patchFrom) chooseMode("zero");
  }, [mode, control?.patchFrom]);
  // A sweep already made here shows again, here or in a what-if run.
  useEffect(() => {
    const last = chosen.get(place)?.sweep;
    const asked = last === key ? sweeps.get(key) : undefined;
    if (!asked) return;
    let live = true;
    void asked.then((result) => live && setSwept({ key, result }));
    return () => {
      live = false;
    };
  }, [place, key]);
  if (!control || !op) return null;
  const step = op.index;
  const knocked =
    control.knocked?.step === step && (control.knocked.output ?? 0) === output
      ? control.knocked
      : null;
  const label = (at: number) => {
    const words = tokens?.(tensor, at);
    const index = coords[at] ?? 0;
    const word = words && !words.vocabulary ? words.words[index] : undefined;
    return `${axisName(tensor, at) ?? `axis ${at}`} = ${index}${word ? ` (${word})` : ""}`;
  };
  const rule = (index: number | null, along = axis): Knockout => ({
    step,
    output,
    mode,
    axis: index === null ? null : along,
    index,
    patch_from: mode === "patch" ? control.patchFrom : null,
  });
  const sweepable =
    axis !== null &&
    tensor.shape[axis] >= 2 &&
    tensor.shape[axis] <= 64 &&
    (mode !== "patch" || !!control.patchFrom);
  const sweep = () => {
    if (axis === null) return;
    let asked = sweeps.get(key);
    if (!asked) {
      asked = control.sweep({
        step,
        output,
        axis,
        mode,
        patch_from: mode === "patch" ? control.patchFrom : null,
      });
      asked.then((found) => !found && sweeps.delete(key));
      sweeps.set(key, asked);
    }
    setSwept({ key });
    remember({ sweep: key });
    void asked.then((result) =>
      setSwept((current) => (current?.key === key ? { key, result } : current)),
    );
  };
  const shown = swept?.key === key ? swept : null;
  const result = shown?.result;
  const effects = result?.effects ?? [];
  const cellValues = result?.cell_values ?? [];
  const largest = Math.max(0, ...effects.map((effect) => effect ?? 0));
  const sliceName = (index: number) => {
    const along = result?.axis ?? axis;
    if (along === null) return `${index}`;
    const words = tokens?.(tensor, along);
    const word = words && !words.vocabulary ? words.words[index] : undefined;
    return word ? `${index} ${word}` : `${index}`;
  };
  return (
    <div
      className="what-if knockout"
      title="Run the same code with this step's result replaced as it is made; nothing is saved"
    >
      <select
        aria-label="Which part to knock out"
        value={axis ?? ""}
        disabled={control.busy}
        onChange={(event) =>
          setAxis(event.target.value === "" ? null : Number(event.target.value))
        }
      >
        <option value="">all of {tensor.name}</option>
        {tensor.shape.map((size, at) =>
          size > 1 ? (
            <option key={at} value={at}>
              only {label(at)}
            </option>
          ) : null,
        )}
      </select>
      <select
        aria-label="What to put in"
        value={mode}
        disabled={control.busy}
        onChange={(event) => setMode(event.target.value as Mode)}
      >
        <option value="zero">to 0</option>
        <option value="mean" disabled={!tensor.dtype.startsWith("float")}>
          {axis === null ? "to its mean" : "to the mean slice"}
        </option>
        <option value="patch" disabled={!control.patchFrom}>
          from the run before
        </option>
      </select>
      <button
        type="button"
        disabled={control.busy}
        onClick={() =>
          control.knockout(rule(axis === null ? null : (coords[axis] ?? 0)))
        }
        title="Run again with this replaced; Diff and the change lens show what it changed"
      >
        Try
      </button>
      <button
        type="button"
        disabled={control.busy || !sweepable}
        onClick={sweep}
        title={
          axis === null
            ? "Choose an axis to sweep: each of its slices is knocked out in turn"
            : "Knock out each slice along this axis in turn, and measure how far the output moves"
        }
      >
        Sweep{" "}
        {axis === null ? "an axis" : (axisName(tensor, axis) ?? `axis ${axis}`)}
      </button>
      {knocked && control.leave && (
        <button
          type="button"
          onClick={() => control.knockout(null)}
          title="Run without the knockout"
        >
          Undo knockout
        </button>
      )}
      {shown && result === undefined && (
        <span className="knockout-status">Sweeping…</span>
      )}
      {result?.error && (
        <span className="knockout-status knockout-error">
          {result.error.message}
        </span>
      )}
      {result && !result.error && (
        <div
          className="knockout-sweep"
          aria-label="How far the output moved with each slice knocked out"
        >
          {effects.map((effect, index) => (
            <button
              type="button"
              key={index}
              className={
                effect !== null && effect === largest && largest > 0
                  ? "knockout-bar knockout-most"
                  : "knockout-bar"
              }
              disabled={control.busy}
              onClick={() => control.knockout(rule(index, result.axis))}
              title={`Output moved ${effect === null ? "by an unknown amount" : `by ${percent(effect)} (‖y − y₀‖ / ‖y₀‖)`}${result.cell_value != null && cellValues[index] != null ? `; its largest value went ${formatValue(result.cell_value)} → ${formatValue(cellValues[index]!)}` : ""}. Click to run this knockout.`}
            >
              <span className="knockout-slice">{sliceName(index)}</span>
              <span className="knockout-track">
                <span
                  style={{
                    width: `${largest > 0 && effect ? (effect / largest) * 100 : 0}%`,
                  }}
                />
              </span>
              <span className="knockout-effect">
                {effect === null ? "—" : percent(effect)}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

import { useState } from "react";
import { ArrowRight, MousePointer2 } from "lucide-react";
import type { Run } from "../api/client";
import { TensorCard } from "../tensors/TensorCard";
import { IndexControl } from "../tensors/TensorNavigation";
import { ValuesToggle } from "../tensors/ValuesToggle";
import { useTensorValues } from "../tensors/useTensorValues";
import { formatCellValue } from "../tensors/coordinates";
import { reductionNumbers } from "./reductionStatistics";
import { useReductionStatistics } from "./useReductionStatistics";
import {
  reductionInputSelection,
  reductionReference,
  reductionSelection,
  reductionWindow,
  type Reduction,
} from "./reduction";
import { useCellPaint } from "../tensors/InkShape";
import "../tensors/gridInk.css";

const coord = (a: number[]) => `[${a.join(", ")}]`;
export function ReductionView({
  reduction: p,
  run,
  showValues,
  onShowValues,
  onDetails,
}: {
  reduction: Reduction;
  run: Run;
  showValues: boolean;
  onShowValues: (value: boolean) => void;
  onDetails: () => void;
}) {
  const [cursor, setCursor] = useState({ output: 0, term: 0 });
  const selection = reductionSelection(p, cursor.output, cursor.term);
  const window = reductionWindow(p, cursor.output, cursor.term);
  const samples = reductionWindow(p, cursor.output, cursor.term, true);
  const complete = samples.length === p.size;
  const shapeOnly =
    p.input.value_source === "shape" || p.output.value_source === "shape";
  const numeric = showValues && !shapeOnly;
  const summary = useReductionStatistics(
    {
      runId: run.id,
      operationId: p.operationId,
      tensorId: p.input.id,
      outputIndex: cursor.output,
      size: p.size,
      kind: p.kind,
      integer: !p.input.dtype.includes("float"),
    },
    numeric && !complete && !p.cast,
  );
  const paint = useCellPaint();
  const inputData = useTensorValues(
    p.input,
    run.id,
    numeric ? samples.map((s) => s.input) : [],
  );
  const outputData = useTensorValues(
    p.output,
    run.id,
    numeric ? [cursor.output] : [],
  );
  const datasets = [inputData, outputData, summary];
  const error = datasets.find((d) => d.error)?.error;
  const loading = numeric && datasets.some((d) => d.loading);
  const reference = numeric
    ? !complete
      ? reductionNumbers(summary.data)
      : reductionReference(
          p,
          samples.map((s) => inputData.valueAt(s.input)),
        )
    : null;
  const y = numeric ? outputData.valueAt(cursor.output) : undefined;
  const setTerm = (term: number) => setCursor({ ...cursor, term });
  const selectOutput = (output: number) => setCursor({ ...cursor, output });
  const frame = {
    rows: Math.max(
      1,
      ...[p.input, p.output].map((t) => Math.min(8, t.shape.at(-2) ?? 1)),
    ),
    columns: Math.max(
      1,
      ...[p.input, p.output].map((t) => Math.min(8, t.shape.at(-1) ?? 1)),
    ),
  };
  const axisName = (i: number) => p.input.axes[i] || `axis ${i}`;
  const name = p.kind === "mean" ? "Mean" : "Sum";
  const value = (n: number | string | undefined) =>
    numeric && (
      <strong title={String(n ?? "unavailable")}>{String(n ?? "—")}</strong>
    );
  return (
    <section
      className="linear-lesson reduction-lesson"
      aria-label={`${name} reduction lesson`}
    >
      <header className="linear-heading">
        <div>
          <code>{p.kind === "mean" ? "y = Σ xᵢ / N" : "y = Σ xᵢ"}</code>
          <p>
            Keep the other coordinates fixed.{" "}
            {p.kind === "mean" ? "Average" : "Add"} every cell along the reduced
            axes to form one output.
          </p>
        </div>
        <button className="secondary-button" onClick={onDetails}>
          Tensor details <ArrowRight size={13} />
        </button>
      </header>
      <div
        className="reduction-shapes"
        aria-label="Reduction shape progression"
      >
        <div>
          <span className="eyebrow">Input axes</span>
          <div className="reduction-axes">
            {p.input.shape.map((n, i) => (
              <span className={p.axes.includes(i) ? "reduced" : ""} key={i}>
                <b>{n.toLocaleString()}</b>
                <small>{axisName(i)}</small>
                <em>{p.axes.includes(i) ? "reduce" : "keep"}</em>
              </span>
            ))}
            {!p.input.shape.length && (
              <span>
                <b>scalar</b>
                <small>one cell</small>
              </span>
            )}
          </div>
        </div>
        <ArrowRight size={20} />
        <div>
          <span className="eyebrow">Output {coord(p.output.shape)}</span>
          <p>
            {!p.input.shape.length
              ? "A scalar contributes its one value."
              : p.keepdim
                ? "Reduced axes stay in place, with size 1."
                : "Reduced axes disappear; the others keep their order."}
          </p>
          <small>
            {p.size.toLocaleString()} {p.size === 1 ? "cell" : "cells"} → 1
            output · {p.output.numel.toLocaleString()} independent{" "}
            {p.output.numel === 1 ? "group" : "groups"}
          </small>
        </div>
      </div>
      <div className="normalization-controls">
        <IndexControl
          label="Output cell"
          name="Reduction output"
          value={cursor.output}
          size={p.output.numel}
          onChange={selectOutput}
        />
        <IndexControl
          label="Contributor"
          name="Reduction contributor"
          value={cursor.term}
          size={p.size}
          onChange={setTerm}
        />
        <ValuesToggle
          checked={showValues}
          onChange={onShowValues}
          shapeOnly={shapeOnly}
        />
      </div>
      <div className="normalization-location" aria-live="polite">
        <code>
          x{coord(selection.inputCoordinates)} → y
          {coord(selection.outputCoordinates)}
        </code>
        <span>
          {p.axes.length
            ? `Reduced axes: ${p.axes.map((i) => `${axisName(i)} (axis ${i})`).join(", ")}`
            : "Scalar reduction"}{" "}
          · keepdim = {String(p.keepdim)}
        </span>
        <small>
          {p.input.shape.some((_, i) => !p.axes.includes(i))
            ? `Fixed: ${p.input.shape.flatMap((_, i) => (p.axes.includes(i) ? [] : [`${axisName(i)} = ${selection.inputCoordinates[i]}`])).join(", ")}`
            : "All input cells belong to this single output."}
        </small>
      </div>
      <div className="normalization-tensors">
        <TensorCard
          tensor={p.input}
          label="Reduction input"
          runId={run.id}
          gridFrame={frame}
          showValues={showValues}
          focusIndex={selection.input}
          highlights={samples.map((s) => s.input)}
          onSelect={(index) => setCursor(reductionInputSelection(p, index))}
        />
        <ArrowRight className="normalization-direction" size={20} />
        <TensorCard
          tensor={p.output}
          label="Reduced output"
          tone="output"
          runId={run.id}
          gridFrame={frame}
          showValues={showValues}
          focusIndex={cursor.output}
          highlights={[cursor.output]}
          onSelect={selectOutput}
        />
      </div>
      <div className="linear-selection-note">
        <MousePointer2 size={14} />
        <p>
          Select either tensor, including its enlarged 3D view. Highlights show{" "}
          {complete
            ? "all contributors"
            : `${samples.length} contributors in the selected window`}{" "}
          across the current slices. Every output uses all{" "}
          {p.size.toLocaleString()} cells in its group.
        </p>
      </div>
      <section
        className="linear-calculation"
        aria-label="Reduction calculation"
      >
        <header>
          <div>
            <span className="eyebrow">Reference calculation</span>
            <code>y{coord(selection.outputCoordinates)}</code>
          </div>
          <span>
            {name} · {p.size.toLocaleString()} contributors
          </span>
        </header>
        {!complete && numeric && !p.cast && (
          <p className="normalization-coverage" role="status">
            {summary.loading
              ? `Calculating across all ${p.size.toLocaleString()} contributors…`
              : reference
                ? `Calculation covers all ${p.size.toLocaleString()} contributors. The visible window shows ${window.length}.`
                : "The complete-group reference calculation is unavailable."}
          </p>
        )}
        <div
          className="reduction-group"
          aria-label="Selected contributor window"
        >
          <div className="reduction-group-heading">
            <span className="eyebrow">01 · Gather the group</span>
            <small>
              Contributors {window[0].term}–{window.at(-1)!.term} of{" "}
              {p.size.toLocaleString()}
            </small>
          </div>
          <div className="reduction-cells">
            {window[0].term > 0 && (
              <span className="reduction-gap">
                …<small>{window[0].term.toLocaleString()} earlier</small>
              </span>
            )}
            {window.map((s) => (
              <button
                key={s.term}
                className={
                  s.term !== cursor.term && paint?.(p.input, s.input)
                    ? "cell-inked"
                    : undefined
                }
                style={
                  s.term !== cursor.term && paint?.(p.input, s.input)
                    ? ({
                        "--cell-ink": paint(p.input, s.input),
                      } as React.CSSProperties)
                    : undefined
                }
                aria-label={`Inspect reduction contributor ${s.term}`}
                aria-pressed={s.term === cursor.term}
                onClick={() => setTerm(s.term)}
              >
                <small>x{coord(s.inputCoordinates)}</small>
                <b
                  title={
                    numeric
                      ? String(inputData.valueAt(s.input) ?? "unavailable")
                      : undefined
                  }
                >
                  {numeric
                    ? formatCellValue(inputData.valueAt(s.input), 9)
                    : `#${s.term}`}
                </b>
              </button>
            ))}
            {window.at(-1)!.term < p.size - 1 && (
              <span className="reduction-gap">
                …
                <small>
                  {(p.size - window.at(-1)!.term - 1).toLocaleString()} later
                </small>
              </span>
            )}
          </div>
          <p>
            These are cells from the recorded input, arranged here in logical
            axis order.{" "}
            {p.size > window.length
              ? "The window is only part of the full reduction group."
              : "The whole group is shown."}
          </p>
        </div>
        <ol className="reduction-equation">
          <li>
            <span className="eyebrow">02 · Add the group</span>
            <code>
              S ={" "}
              {p.size <= 3
                ? ["x₀", "x₁", "x₂"].slice(0, p.size).join(" + ")
                : "x₀ + x₁ + … + xₙ₋₁"}
            </code>
            {value(reference?.sum)}
            <p>Sum all {p.size.toLocaleString()} contributors.</p>
          </li>
          <li>
            <span className="eyebrow">
              03 ·{" "}
              {p.kind === "mean" ? "Divide by the count" : "Return the sum"}
            </span>
            <code>
              {p.kind === "mean"
                ? `y = S / ${p.size.toLocaleString()}`
                : "y = S"}
            </code>
            {value(reference?.result)}
            <p>
              {p.kind === "mean"
                ? "The divisor counts every cell along every reduced axis."
                : "Sum does not divide by the number of cells."}
            </p>
          </li>
        </ol>
        <div className="normalization-recorded">
          <div>
            <span className="eyebrow">Recorded PyTorch output</span>
            <code>
              y{coord(selection.outputCoordinates)} · {p.output.dtype}
            </code>
          </div>
          {numeric ? (
            value(y)
          ) : (
            <span>
              {shapeOnly ? "Shapes only · no numeric values" : "Values hidden"}
            </span>
          )}
        </div>
        <p className="linear-numeric-note" role="status">
          {shapeOnly
            ? "This run records group membership and output shapes without numeric values."
            : !showValues
              ? "Show values to inspect the calculation."
              : loading
                ? "Loading the complete calculation…"
                : p.cast
                  ? `PyTorch casts ${p.input.dtype} to ${p.output.dtype} before reduction. The captured input is shown before that cast; reference arithmetic is omitted.`
                  : summary.data?.status === "non_finite"
                    ? "The full group contains a non-finite value, possibly outside this window. Inspect the recorded output."
                    : summary.data?.status === "overflow"
                      ? "The full-group reference overflows floating-point arithmetic. Inspect the recorded output."
                      : !reference
                        ? "Some group values are unavailable, non-finite, or overflow the reference calculation. Inspect the recorded output."
                        : reference.sum === undefined
                          ? "The full sum overflows floating-point arithmetic, but the mean is finite. The reference scales each contributor before summing. The recorded output is authoritative."
                          : "The reference shows mathematical arithmetic from captured values. PyTorch's dtype, rounding, accumulation order, and integer overflow can change the result; the recorded output is authoritative."}
        </p>
        {numeric && error && (
          <div className="linear-load-error" role="alert">
            {error}{" "}
            <button
              className="text-button"
              onClick={() => datasets.forEach((d) => d.retry())}
            >
              Retry calculation
            </button>
          </div>
        )}
        <details className="linear-terms">
          <summary>
            Inspect contributor coordinates{" "}
            <span>
              {window[0].term}–{window.at(-1)!.term} / {p.size.toLocaleString()}
            </span>
          </summary>
          <div className="linear-term-navigation">
            <p>Navigate any contributor without loading the whole tensor.</p>
            <IndexControl
              label="Group window"
              name="Reduction group window"
              value={window[0].term}
              size={p.size}
              step={8}
              onChange={setTerm}
            />
          </div>
          <div className="linear-term-table">
            <table>
              <thead>
                <tr>
                  <th>Contributor</th>
                  <th>Input coordinate</th>
                  <th>Input value</th>
                  <th>Output coordinate</th>
                </tr>
              </thead>
              <tbody>
                {window.map((s) => (
                  <tr
                    key={s.term}
                    className={s.term === cursor.term ? "selected" : ""}
                  >
                    <th scope="row">
                      <button
                        aria-label={`Inspect reduction term ${s.term}`}
                        aria-pressed={s.term === cursor.term}
                        onClick={() => setTerm(s.term)}
                      >
                        {s.term}
                      </button>
                    </th>
                    <td>{coord(s.inputCoordinates)}</td>
                    <td
                      title={
                        numeric
                          ? String(inputData.valueAt(s.input) ?? "unavailable")
                          : undefined
                      }
                    >
                      {numeric
                        ? formatCellValue(inputData.valueAt(s.input), 12)
                        : "—"}
                    </td>
                    <td>{coord(s.outputCoordinates)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </section>
    </section>
  );
}

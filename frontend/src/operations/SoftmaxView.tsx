import { useState } from "react";
import { ArrowRight, MousePointer2 } from "lucide-react";
import type { Run } from "../api/client";
import { TensorCard } from "../tensors/TensorCard";
import { IndexControl } from "../tensors/TensorNavigation";
import { ValuesToggle } from "../tensors/ValuesToggle";
import { useTensorValues } from "../tensors/useTensorValues";
import { unravel } from "../tensors/coordinates";
import {
  softmaxCalculation,
  softmaxIndex,
  softmaxSelection,
  softmaxSummary,
  softmaxWindow,
  type Softmax,
} from "./softmax";
import { useSoftmaxStatistics } from "./useSoftmaxStatistics";

const coord = (a: number[]) => `[${a.join(", ")}]`;
const display = (n: unknown) => {
  if (n == null) return "—";
  if (n === -Infinity || n === "-inf") return "−∞";
  return typeof n === "number" && Number.isFinite(n) && String(n).length > 14
    ? Number(n.toPrecision(8)).toString()
    : String(n);
};

export function SoftmaxView({
  softmax: p,
  run,
  showValues,
  onShowValues,
  onDetails,
}: {
  softmax: Softmax;
  run: Run;
  showValues: boolean;
  onShowValues: (value: boolean) => void;
  onDetails: () => void;
}) {
  const [selected, setSelected] = useState(0);
  const { group, member, coordinates } = softmaxSelection(p, selected);
  const window = softmaxWindow(p, selected),
    samples = softmaxWindow(p, selected, true);
  const complete = samples.length === p.size;
  const shapeOnly =
    p.input.value_source === "shape" || p.output.value_source === "shape";
  const numeric = showValues && !shapeOnly;
  const inputData = useTensorValues(
    p.input,
    run.id,
    numeric ? samples.map((s) => s.index) : [],
  );
  const outputData = useTensorValues(
    p.output,
    run.id,
    numeric ? window.map((s) => s.index) : [],
  );
  const remote = useSoftmaxStatistics(
    {
      runId: run.id,
      operationId: p.operationId,
      tensorId: p.input.id,
      group,
      size: p.size,
    },
    numeric && !complete,
  );
  const summary = numeric
    ? complete
      ? softmaxSummary(
          samples.map((s) => inputData.valueAt(s.index)),
          p.size,
        )
      : remote.data
    : null;
  const calculation = softmaxCalculation(summary, inputData.valueAt(selected));
  const datasets = [inputData, outputData, remote],
    error = datasets.find((d) => d.error)?.error;
  const loading = numeric && datasets.some((d) => d.loading);
  const axisName = p.input.axes[p.axis] || `axis ${p.axis}`;
  const value = (n: number | string | null | undefined) =>
    numeric && (
      <strong title={String(n ?? "unavailable")}>
        {n === -Infinity ? "−∞" : String(n ?? "—")}
      </strong>
    );
  const frame = {
    rows: Math.min(8, p.input.shape.at(-2) ?? 1),
    columns: Math.min(8, p.input.shape.at(-1) ?? 1),
  };
  const changeGroup = (next: number) =>
    setSelected(softmaxIndex(p, next, member));
  const changeMember = (next: number) =>
    setSelected(softmaxIndex(p, group, next));
  return (
    <section
      className="linear-lesson softmax-lesson"
      aria-label="Softmax lesson"
    >
      <header className="linear-heading">
        <div>
          <code>yᵢ = exp(xᵢ − m) / Σⱼ exp(xⱼ − m)</code>
          <p>
            One shared denominator per group. Shape and coordinates stay the
            same; scores become weights.
          </p>
        </div>
        <button className="secondary-button" onClick={onDetails}>
          Tensor details <ArrowRight size={13} />
        </button>
      </header>
      <div
        className="reduction-shapes"
        role="group"
        aria-label="Softmax axis groups"
      >
        <div>
          <span className="eyebrow">Normalize one axis</span>
          <div className="reduction-axes">
            {p.input.shape.map((n, i) => (
              <span key={i} className={i === p.axis ? "reduced" : ""}>
                <b>{n.toLocaleString()}</b>
                <small>{p.input.axes[i] || `axis ${i}`}</small>
                <em>{i === p.axis ? "normalize" : "fixed"}</em>
              </span>
            ))}
            {!p.input.shape.length && (
              <span>
                <b>scalar</b>
                <small>one score</small>
              </span>
            )}
          </div>
        </div>
        <ArrowRight size={20} />
        <div>
          <span className="eyebrow">Output {coord(p.output.shape)}</span>
          <p>
            {p.input.shape.length
              ? `Each group varies along ${axisName}; all other coordinates stay fixed.`
              : "A finite scalar becomes weight 1."}
          </p>
          <small>
            {p.size.toLocaleString()} {p.size === 1 ? "score" : "scores"} per
            group · {p.groups.toLocaleString()} independent{" "}
            {p.groups === 1 ? "group" : "groups"}
          </small>
        </div>
      </div>
      <div className="normalization-controls">
        <IndexControl
          label="Group"
          name="Softmax group"
          value={group}
          size={p.groups}
          onChange={changeGroup}
        />
        <IndexControl
          label="Score"
          name="Softmax score"
          value={member}
          size={p.size}
          onChange={changeMember}
        />
        <ValuesToggle
          checked={showValues}
          onChange={onShowValues}
          shapeOnly={shapeOnly}
        />
      </div>
      <div className="normalization-location" aria-live="polite">
        <code>
          x{coord(coordinates)} → y{coord(coordinates)}
        </code>
        <span>
          {p.input.shape.length
            ? `Normalize ${axisName} (axis ${p.axis})`
            : "Scalar softmax"}{" "}
          · group {group.toLocaleString()}
        </span>
        <small>
          {p.input.shape.length > 1
            ? `Fixed: ${coordinates.flatMap((n, i) => (i === p.axis ? [] : [`${p.input.axes[i] || `axis ${i}`} = ${n}`])).join(", ")}`
            : "All cells share this denominator."}
        </small>
      </div>
      <div className="normalization-tensors">
        <TensorCard
          tensor={p.input}
          label="Softmax scores"
          runId={run.id}
          gridFrame={frame}
          showValues={showValues}
          focusIndex={selected}
          highlights={samples.map((s) => s.index)}
          onSelect={setSelected}
        />
        <ArrowRight className="normalization-direction" size={20} />
        <TensorCard
          tensor={p.output}
          label="Softmax weights"
          tone="output"
          runId={run.id}
          gridFrame={frame}
          showValues={showValues}
          focusIndex={selected}
          highlights={samples.map((s) => s.index)}
          onSelect={setSelected}
        />
      </div>
      <div className="linear-selection-note">
        <MousePointer2 size={14} />
        <p>
          Select either tensor, including its enlarged 3D view. Highlights
          follow{" "}
          {complete
            ? "the complete group"
            : `${samples.length} scores in the selected window`}
          . The denominator always uses all {p.size.toLocaleString()} scores.
        </p>
      </div>
      <section className="linear-calculation" aria-label="Softmax calculation">
        <header>
          <div>
            <span className="eyebrow">Reference calculation</span>
            <code>
              x{coord(coordinates)} → y{coord(coordinates)}
            </code>
          </div>
          <span>
            Group {group.toLocaleString()} · {p.size.toLocaleString()} scores
          </span>
        </header>
        {numeric && (
          <p className="normalization-coverage" role="status">
            {loading
              ? `Loading the complete group of ${p.size.toLocaleString()} scores…`
              : summary?.status === "ok"
                ? `Maximum and denominator cover all ${p.size.toLocaleString()} scores. ${summary.masked_count ? `${summary.masked_count.toLocaleString()} negative-infinity scores contribute zero.` : "Every finite group has at least one exponential equal to 1."}`
                : "A finite reference is unavailable for this group."}
          </p>
        )}
        <ol className="reduction-equation softmax-equation">
          <li>
            <span className="eyebrow">01 · Find the maximum</span>
            <code>m = max(x₀, …, xₙ₋₁)</code>
            {value(summary?.maximum)}
            <p>Use the largest score in the whole group.</p>
          </li>
          <li>
            <span className="eyebrow">02 · Shift the score</span>
            <code>zᵢ = xᵢ − m</code>
            {value(calculation?.shifted)}
            <p>
              The maximum becomes zero. Relative score differences stay the
              same.
            </p>
          </li>
          <li>
            <span className="eyebrow">03 · Exponentiate</span>
            <code>eᵢ = exp(zᵢ)</code>
            {value(calculation?.exponential)}
            <p>
              Shifted scores are nonpositive, so exponentials stay between zero
              and one.
            </p>
          </li>
          <li>
            <span className="eyebrow">04 · Normalize</span>
            <code>yᵢ = eᵢ / Z</code>
            {value(calculation?.weight)}
            <p>
              <span
                title={
                  numeric
                    ? String(summary?.denominator ?? "unavailable")
                    : undefined
                }
              >
                Z = Σⱼ eⱼ
                {numeric ? ` = ${String(summary?.denominator ?? "—")}` : ""}
              </span>
              . Every score shares this total.
            </p>
          </li>
        </ol>
        <div className="normalization-recorded">
          <div>
            <span className="eyebrow">Recorded PyTorch output</span>
            <code>
              y{coord(coordinates)} · {p.output.dtype}
            </code>
          </div>
          {numeric ? (
            value(outputData.valueAt(selected))
          ) : (
            <span>
              {shapeOnly ? "Shapes only · no numeric values" : "Values hidden"}
            </span>
          )}
        </div>
        <p className="linear-numeric-note" role="status">
          {shapeOnly
            ? "This run captured shapes only. Groups and coordinates remain interactive."
            : !showValues
              ? "Show values to inspect the calculation."
              : loading
                ? "Loading captured scores, weights, and full-group statistics…"
                : summary?.status === "all_masked"
                  ? "Every score is negative infinity. There is no finite maximum to subtract: this group has no finite softmax distribution. The recorded output is shown without inventing zero weights."
                  : summary?.status === "non_finite"
                    ? "This group contains NaN or positive infinity, possibly outside the visible window. The finite reference formula is unavailable; inspect the recorded output."
                    : !calculation
                      ? "Some captured values are unavailable. No partial denominator is used."
                      : "Reference arithmetic uses the saved scores. PyTorch rounding and underflow can differ; the recorded weights are authoritative. Weights sum to one per valid group, up to rounding."}
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
        <div className="softmax-window">
          <header>
            <div>
              <span className="eyebrow">Scores → weights</span>
              <p>
                {window.length === p.size
                  ? "The complete group is shown."
                  : `Window ${window[0].member}–${window.at(-1)!.member} of ${p.size.toLocaleString()} scores.`}
                {numeric &&
                  " Bars use a fixed 0–1 scale. Table values are rounded; hover for full precision."}
              </p>
            </div>
            <IndexControl
              label="Window"
              name="Softmax score window"
              value={window[0].member}
              size={p.size}
              step={8}
              onChange={changeMember}
            />
          </header>
          <div className="linear-term-table">
            <table>
              <thead>
                <tr>
                  <th>Coordinate</th>
                  <th>Score xᵢ</th>
                  <th>Shift xᵢ − m</th>
                  <th>exp(xᵢ − m)</th>
                  <th>Recorded weight</th>
                </tr>
              </thead>
              <tbody>
                {window.map(({ index, member: term }) => {
                  const x = numeric ? inputData.valueAt(index) : undefined,
                    y = numeric ? outputData.valueAt(index) : undefined,
                    calc = softmaxCalculation(summary, x);
                  return (
                    <tr
                      key={index}
                      className={index === selected ? "selected" : ""}
                    >
                      <th scope="row">
                        <button
                          aria-label={`Inspect softmax score ${term}`}
                          aria-pressed={index === selected}
                          onClick={() => setSelected(index)}
                        >
                          {coord(unravel(index, p.input.shape))}
                        </button>
                      </th>
                      <td title={String(x ?? "")}>
                        {numeric ? display(x) : "—"}
                      </td>
                      <td title={String(calc?.shifted ?? "")}>
                        {numeric ? display(calc?.shifted) : "—"}
                      </td>
                      <td title={String(calc?.exponential ?? "")}>
                        {numeric ? display(calc?.exponential) : "—"}
                      </td>
                      <td title={String(y ?? "")}>
                        <span>{numeric ? display(y) : "—"}</span>
                        {numeric &&
                          typeof y === "number" &&
                          Number.isFinite(y) &&
                          y >= 0 &&
                          y <= 1 && (
                            <span
                              className="softmax-weight-bar"
                              aria-hidden="true"
                            >
                              <i style={{ width: `${y * 100}%` }} />
                            </span>
                          )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </section>
  );
}

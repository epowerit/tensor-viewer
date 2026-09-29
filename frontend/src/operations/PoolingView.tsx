import { useState } from "react";
import { ArrowRight, MousePointer2 } from "lucide-react";
import type { Run } from "../api/client";
import { TensorCard } from "../tensors/TensorCard";
import { IndexControl } from "../tensors/TensorNavigation";
import { ValuesToggle } from "../tensors/ValuesToggle";
import { useTensorValues } from "../tensors/useTensorValues";
import { formatValue, unravel } from "../tensors/coordinates";
import {
  poolingRegion,
  poolingSample,
  poolingSamples,
  poolingInputSelection,
  poolingCalculation,
  poolingWinner,
  type Pooling,
} from "./pooling";

const coord = (a: number[]) => `[${a.join(", ")}]`;
export function PoolingView({
  pooling: p,
  run,
  showValues,
  onShowValues,
  onDetails,
}: {
  pooling: Pooling;
  run: Run;
  showValues: boolean;
  onShowValues: (show: boolean) => void;
  onDetails: () => void;
}) {
  const [cursor, setCursor] = useState({ output: 0, term: 0 });
  const [unmatched, setUnmatched] = useState<number | null>(null);
  const region = poolingRegion(p, cursor.output);
  const selected = poolingSample(p, region, cursor.term);
  const samples = poolingSamples(p, region, cursor.term, true),
    window = poolingSamples(p, region, cursor.term);
  const shapeOnly = p.input.value_source === "shape",
    numeric = showValues && !shapeOnly;
  const inputData = useTensorValues(
    p.input,
    run.id,
    numeric ? samples.flatMap((s) => (s.input === null ? [] : [s.input])) : [],
  );
  const outputData = useTensorValues(
    p.output,
    run.id,
    numeric ? [cursor.output] : [],
  );
  const indexData = useTensorValues(
    p.indices ?? p.output,
    run.id,
    numeric && p.indices ? [cursor.output] : [],
  );
  const datasets = [inputData, outputData, ...(p.indices ? [indexData] : [])];
  const error = datasets.find((d) => d.error)?.error;
  const recorded = outputData.valueAt(cursor.output);
  const index = numeric ? indexData.valueAt(cursor.output) : undefined;
  const winner = numeric ? poolingWinner(p, region, index) : null;
  const calculation = poolingCalculation(
    p,
    region,
    samples,
    inputData.valueAt,
    recorded,
  );
  const update = (next: typeof cursor) => {
    setCursor(next);
    setUnmatched(null);
  };
  const selectOutput = (output: number) =>
    update({
      output,
      term: Math.min(cursor.term, poolingRegion(p, output).terms - 1),
    });
  const selectTerm = (term: number) => update({ ...cursor, term });
  const reference = `y${coord(region.coordinates)}`;
  const inputHighlights = samples.flatMap((s) =>
    s.input === null ? [] : [s.input],
  );
  const frame = {
    rows: Math.max(
      ...[p.input, p.output].map((t) => Math.min(8, t.shape.at(-2)!)),
    ),
    columns: Math.max(
      ...[p.input, p.output].map((t) => Math.min(8, t.shape.at(-1)!)),
    ),
  };
  const paddingValue = p.mode === "max" ? "−∞" : "0";
  return (
    <section className="pooling-lesson" aria-label="Pooling lesson">
      <header className="pooling-summary">
        <p>
          {p.mode === "adaptive"
            ? "Each output selects an input region from their size ratio. Neighboring regions can overlap."
            : p.mode === "max"
              ? "Keep the largest value in each window. Padded samples are negative infinity, so they cannot beat a finite input."
              : "Add the real input values. Padding contributes zero; the divisor determines how padding is counted."}
        </p>
        <button className="secondary-button" onClick={onDetails}>
          Tensor details <ArrowRight size={13} />
        </button>
      </header>
      <div className="pooling-geometry" aria-label="Recorded pooling geometry">
        {p.mode !== "adaptive" ? (
          <>
            <span>
              Kernel <b>{p.kernel.join(" × ")}</b>
            </span>
            <span>
              Stride <b>{coord(p.stride)}</b>
            </span>
            <span>
              Padding <b>{coord(p.padding)}</b>
            </span>
            {p.mode === "max" && (
              <span>
                Dilation <b>{coord(p.dilation)}</b>
              </span>
            )}
            <span>
              Ceil mode <b>{p.ceil ? "On" : "Off"}</b>
            </span>
          </>
        ) : (
          <span>
            Adaptive output <b>{p.output.shape.slice(-2).join(" × ")}</b>
          </span>
        )}
        <span>Batch and channel are unchanged</span>
      </div>
      <div className="pooling-controls">
        <IndexControl
          label="Window sample"
          name="Pooling window sample"
          value={cursor.term}
          size={region.terms}
          onChange={selectTerm}
        />
        <ValuesToggle
          checked={showValues}
          onChange={onShowValues}
          shapeOnly={shapeOnly}
        />
      </div>
      <div className="pooling-location" aria-live="polite">
        <code>
          {selected.padding
            ? `Virtual padding ${paddingValue} at ${coord(selected.coordinates)}`
            : `x${coord(selected.coordinates)}`}{" "}
          · window for {reference}
        </code>
        <span>
          {region.real.toLocaleString()} real samples
          {region.padded
            ? ` + ${region.padded.toLocaleString()} padded samples`
            : ""}
          {region.divisor !== null
            ? ` · divisor ${region.divisor.toLocaleString()}`
            : ""}
        </span>
        {p.mode === "average" && (
          <small>
            {p.divisor !== null
              ? "The explicit divisor replaces the window count."
              : p.includePadding
                ? "Declared zero padding counts in the divisor; ceil overhang beyond it does not."
                : "Only real input samples count in the divisor."}
          </small>
        )}
        {unmatched !== null && (
          <p role="status">
            Input {coord(unravel(unmatched, p.input.shape))} is skipped by these
            pooling windows. The selected output is unchanged.
          </p>
        )}
      </div>
      <div className="pooling-tensors">
        <TensorCard
          tensor={p.input}
          label="Pooling input"
          runId={run.id}
          gridFrame={frame}
          showValues={showValues}
          focusIndex={unmatched ?? selected.input ?? undefined}
          highlights={inputHighlights}
          onSelect={(input) => {
            const next = poolingInputSelection(p, input, cursor.output);
            if (next) update(next);
            else setUnmatched(input);
          }}
        />
        <ArrowRight className="pooling-direction" size={20} />
        <TensorCard
          tensor={p.output}
          label="Pooled output"
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
          Select an output to move its window, or an input to find a nearby
          window that includes it. Both explorers support 3D selection.
          Highlights show{" "}
          {calculation.complete
            ? "all real samples"
            : `real samples from the selected ${window.length}-position window`}
          , across the current slices. Indices are zero-based.
        </p>
      </div>
      <section className="linear-calculation" aria-label="Pooling calculation">
        <header>
          <div>
            <span className="eyebrow">How this cell is formed</span>
            <code>{reference}</code>
          </div>
          <span>
            {p.mode === "max" ? "Maximum" : "Sum ÷ divisor"} ·{" "}
            {region.terms.toLocaleString()} sampled positions
          </span>
        </header>
        <div className="pooling-formula">
          <code>
            {p.mode === "adaptive"
              ? "start = floor(output × input size ÷ output size); end = ceil((output + 1) × input size ÷ output size)"
              : "input position = output position × stride − padding + window position × dilation"}
          </code>
          <span>
            Region starts at {coord(region.start)} · {region.shape.join(" × ")}{" "}
            samples
            {p.mode === "adaptive"
              ? " · end coordinates are excluded"
              : ` · sample spacing ${coord(p.dilation)}`}
          </span>
        </div>
        <div className="pooling-equation">
          <div>
            <span>
              {calculation.complete
                ? p.mode === "max"
                  ? "Largest real input"
                  : "Sum of real inputs"
                : p.mode === "max"
                  ? "Maximum in this sample window"
                  : "Sum in this sample window"}
            </span>
            {numeric && <strong>{formatValue(calculation.aggregate)}</strong>}
            {!calculation.complete && (
              <small>
                Partial window. The remaining samples have not been loaded for
                this calculation.
              </small>
            )}
            {region.divisor !== null && (
              <code>
                ÷ {region.divisor.toLocaleString()}
                {p.divisor !== null ? " (override)" : ""}
              </code>
            )}
            {numeric && calculation.result !== undefined && (
              <small>
                Calculated result ≈ {formatValue(calculation.result)}
              </small>
            )}
          </div>
          <div>
            <span>Recorded output</span>
            <code>{reference}</code>
            {numeric && (
              <strong title={String(recorded ?? "unavailable")}>
                {formatValue(recorded)}
              </strong>
            )}
            {shapeOnly && (
              <small>Shapes only · no recorded numeric values</small>
            )}
          </div>
        </div>
        {p.mode === "max" && (
          <div
            className="pooling-winner"
            aria-label="Maximum selection provenance"
          >
            {winner ? (
              <>
                <p>
                  Recorded spatial index <b>{String(index)}</b> identifies input{" "}
                  <code>{coord(winner.coordinates)}</code> in this batch and
                  channel.
                </p>
                <button
                  className="secondary-button"
                  onClick={() => selectTerm(winner.term)}
                >
                  Follow recorded maximum <ArrowRight size={13} />
                </button>
              </>
            ) : p.indices ? (
              <p>
                {shapeOnly
                  ? "This shapes-only run has no numeric winner indices."
                  : !numeric
                    ? "Show values to inspect the recorded winner index."
                    : index === undefined
                      ? "The recorded winner index is loading or unavailable."
                      : "The recorded index does not identify a real cell in this window; no input winner is inferred."}
              </p>
            ) : numeric && calculation.matches.length ? (
              <>
                <p>
                  {calculation.matches.length === 1
                    ? "One input matches the recorded maximum in this complete window."
                    : `${calculation.matches.length} inputs tie for the recorded maximum. This operation did not return a winner index.`}
                </p>
                <button
                  className="secondary-button"
                  onClick={() => {
                    const next = poolingInputSelection(
                      p,
                      calculation.matches[0],
                      cursor.output,
                    );
                    if (next) update(next);
                  }}
                >
                  Inspect{" "}
                  {calculation.matches.length === 1
                    ? "matching input"
                    : "a tied input"}{" "}
                  <ArrowRight size={13} />
                </button>
              </>
            ) : (
              <p>
                No winner index was returned.{" "}
                {calculation.complete
                  ? "Numeric matching requires finite recorded values for the full window."
                  : "Only part of this window is loaded; no global winner is inferred."}
              </p>
            )}
          </div>
        )}
        {numeric && (
          <p className="linear-numeric-note">
            {calculation.aggregate === undefined
              ? "A required value is missing, non-finite, or there are no real input samples. The recorded output remains authoritative."
              : "Calculations use captured values and the recorded geometry. Browser rounding and accumulation can differ from PyTorch."}
          </p>
        )}
        {numeric && error && (
          <div className="linear-load-error" role="alert">
            {error}{" "}
            <button
              className="text-button"
              onClick={() => datasets.forEach((d) => d.retry())}
            >
              Retry values
            </button>
          </div>
        )}
        <details className="linear-terms">
          <summary>
            Inspect window samples{" "}
            <span>
              {window[0].term}–{window.at(-1)!.term} /{" "}
              {region.terms.toLocaleString()}
            </span>
          </summary>
          <div className="linear-term-navigation">
            <p>
              Each row is a sampled position, including virtual padding. It is
              not a separate tensor operation.
            </p>
            <IndexControl
              label="Sample window"
              name="Pooling sample window"
              value={window[0].term}
              size={region.terms}
              step={8}
              onChange={selectTerm}
            />
          </div>
          <div className="linear-term-table">
            <table>
              <thead>
                <tr>
                  <th>Sample</th>
                  <th>Input coordinate</th>
                  <th>{numeric ? "Recorded value" : "Source"}</th>
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
                        aria-label={`Inspect pooling sample ${s.term}`}
                        aria-pressed={s.term === cursor.term}
                        onClick={() => selectTerm(s.term)}
                      >
                        {s.term}
                      </button>
                    </th>
                    <td>{coord(s.coordinates)}</td>
                    <td
                      title={
                        numeric && s.input !== null
                          ? String(inputData.valueAt(s.input) ?? "unavailable")
                          : undefined
                      }
                    >
                      {s.padding
                        ? `${paddingValue} · virtual padding`
                        : numeric
                          ? formatValue(inputData.valueAt(s.input!))
                          : "Input tensor"}
                    </td>
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

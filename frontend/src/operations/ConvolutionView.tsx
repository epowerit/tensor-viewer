import { useState } from "react";
import { ArrowRight, MousePointer2 } from "lucide-react";
import type { Run } from "../api/client";
import { TensorCard } from "../tensors/TensorCard";
import { IndexControl } from "../tensors/TensorNavigation";
import { ValuesToggle } from "../tensors/ValuesToggle";
import { useTensorValues } from "../tensors/useTensorValues";
import { formatValue, ravel, unravel } from "../tensors/coordinates";
import { finiteProduct, contributionSum } from "./linear";
import {
  convolutionSelection,
  convolutionWindow,
  convolutionInputSelection,
  convolutionWeightSelection,
  type Convolution,
} from "./convolution";

const coord = (values: number[]) => `[${values.join(", ")}]`;
export function ConvolutionView({
  convolution: p,
  run,
  showValues,
  onShowValues,
  onDetails,
}: {
  convolution: Convolution;
  run: Run;
  showValues: boolean;
  onShowValues: (show: boolean) => void;
  onDetails: () => void;
}) {
  const [cursor, setCursor] = useState({ output: 0, term: 0 });
  const [unmatched, setUnmatched] = useState<number[] | null>(null);
  const selected = convolutionSelection(p, cursor.output, cursor.term);
  const window = convolutionWindow(p, cursor.output, cursor.term);
  const complete = p.terms <= 256;
  const samples = complete
    ? Array.from({ length: p.terms }, (_, term) =>
        convolutionSelection(p, cursor.output, term),
      )
    : window;
  const shapeOnly = p.input.value_source === "shape",
    numeric = showValues && !shapeOnly;
  const inputIndices = samples.flatMap((s) =>
    s.input === null ? [] : [s.input],
  );
  const inputData = useTensorValues(
    p.input,
    run.id,
    numeric ? inputIndices : [],
  );
  const weightData = useTensorValues(
    p.weight,
    run.id,
    numeric ? samples.map((s) => s.weight) : [],
  );
  const outputData = useTensorValues(
    p.output,
    run.id,
    numeric ? [cursor.output] : [],
  );
  const biasData = useTensorValues(
    p.bias ?? p.weight,
    run.id,
    numeric && p.bias ? [selected.feature] : [],
  );
  const datasets = [
    inputData,
    weightData,
    outputData,
    ...(p.bias ? [biasData] : []),
  ];
  const error = datasets.find((d) => d.error)?.error;
  const xAt = (s: typeof selected) =>
    s.padding ? 0 : inputData.valueAt(s.input!);
  const sum = contributionSum(
    samples.map((s) => finiteProduct(xAt(s), weightData.valueAt(s.weight))),
  );
  const bias = p.bias ? biasData.valueAt(selected.feature) : 0;
  const total =
    complete &&
    sum !== undefined &&
    typeof bias === "number" &&
    Number.isFinite(bias) &&
    Number.isFinite(sum + bias)
      ? sum + bias
      : undefined;
  const update = (next: typeof cursor) => {
    setCursor(next);
    setUnmatched(null);
  };
  const selectTerm = (term: number) => update({ ...cursor, term });
  const frame = {
    rows: Math.max(
      ...[p.input, p.weight, p.output].map((t) =>
        Math.min(8, t.shape.at(-2) ?? 1),
      ),
    ),
    columns: Math.max(
      ...[p.input, p.weight, p.output].map((t) =>
        Math.min(8, t.shape.at(-1) ?? 1),
      ),
    ),
  };
  const spatialNames = p.dimensions === 1 ? ["length"] : ["height", "width"];
  const axes = [
    ...(p.channelAxis ? ["batch"] : []),
    "channels",
    ...spatialNames,
  ];
  const reference = `y${coord(selected.coordinates)}`;
  return (
    <section className="convolution-lesson" aria-label="Convolution lesson">
      <header className="convolution-summary">
        <p>
          Slide the recorded kernel without flipping it. Each output uses only
          the input channels in its group.
        </p>
        <button className="secondary-button" onClick={onDetails}>
          Tensor details <ArrowRight size={13} />
        </button>
      </header>
      <div
        className="convolution-geometry"
        aria-label="Recorded convolution geometry"
      >
        <span>
          Kernel <b>{p.kernel.join(" × ")}</b>
        </span>
        <span>
          Stride <b>{coord(p.stride)}</b>
        </span>
        <span>
          Dilation <b>{coord(p.dilation)}</b>
        </span>
        <span>
          Padding before / after{" "}
          <b>
            {coord(p.before)} / {coord(p.after)}
          </b>
        </span>
        <span>
          Groups <b>{p.groups.toLocaleString()}</b>
        </span>
      </div>
      <div className="convolution-controls">
        <IndexControl
          label="Output channel"
          name="Convolution output channel"
          value={selected.feature}
          size={p.weight.shape[0]}
          onChange={(feature) => {
            const coordinates = [...selected.coordinates];
            coordinates[p.channelAxis] = feature;
            update({ ...cursor, output: ravel(coordinates, p.output.shape) });
          }}
        />
        <IndexControl
          label="Kernel contribution"
          name="Convolution contribution"
          value={cursor.term}
          size={p.terms}
          onChange={selectTerm}
        />
        <ValuesToggle
          checked={showValues}
          onChange={onShowValues}
          shapeOnly={shapeOnly}
        />
      </div>
      <div className="convolution-location" aria-live="polite">
        <code>
          {selected.padding
            ? "Zero padding"
            : `x${coord(selected.inputCoordinates)}`}{" "}
          × W{coord(selected.weightCoordinates)} · contributes to {reference}
        </code>
        <span>
          Group {selected.group} · input channels {selected.group * p.channels}–
          {(selected.group + 1) * p.channels - 1}
        </span>
        {selected.padding && (
          <p>
            The sampled coordinate {coord(selected.inputCoordinates)} is outside
            the input. It reads virtual zero padding; no input cell exists
            there.
          </p>
        )}
        {unmatched && (
          <p role="status">
            Input {coord(unmatched)} is never sampled with this stride and
            dilation. The displayed output contribution is unchanged.
          </p>
        )}
      </div>
      <div className="linear-tensors">
        <TensorCard
          tensor={{ ...p.input, axes }}
          label="Input"
          runId={run.id}
          gridFrame={frame}
          showValues={showValues}
          focusIndex={
            unmatched
              ? ravel(unmatched, p.input.shape)
              : (selected.input ?? undefined)
          }
          highlights={samples.flatMap((s) =>
            s.input === null ? [] : [s.input],
          )}
          onSelect={(input) => {
            const next = convolutionInputSelection(p, input, cursor.output);
            if (next) update(next);
            else setUnmatched(unravel(input, p.input.shape));
          }}
        />
        <TensorCard
          tensor={{
            ...p.weight,
            axes: [
              "out_channels",
              "group_channels",
              ...spatialNames.map((n) => `kernel_${n}`),
            ],
          }}
          label="Kernel"
          runId={run.id}
          gridFrame={frame}
          showValues={showValues}
          focusIndex={selected.weight}
          highlights={samples.map((s) => s.weight)}
          onSelect={(weight) =>
            update(convolutionWeightSelection(p, weight, cursor.output))
          }
        />
        <TensorCard
          tensor={{ ...p.output, axes }}
          label="Output"
          tone="output"
          runId={run.id}
          gridFrame={frame}
          showValues={showValues}
          focusIndex={cursor.output}
          highlights={[cursor.output]}
          onSelect={(output) => update({ ...cursor, output })}
        />
      </div>
      <div className="linear-selection-note">
        <MousePointer2 size={14} />
        <p>
          Select an output, kernel, or input cell, including in 3D. An input
          selects a nearby output that uses it, keeping the current output when
          possible. Highlights show{" "}
          {complete ? "all" : `up to ${window.length} windowed`} contributions;
          only the current tensor slices are visible. Indices are zero-based.
        </p>
      </div>
      <section
        className="linear-calculation"
        aria-label="Convolution calculation"
      >
        <header>
          <div>
            <span className="eyebrow">How this cell is formed</span>
            <code>{reference}</code>
          </div>
          <span>
            {p.terms.toLocaleString()} products
            {p.bias ? " + bias" : " · no bias"}
          </span>
        </header>
        <div className="convolution-formula">
          <code>
            input position = output position × stride − padding before + kernel
            position × dilation
          </code>
          <span>
            {coord(selected.coordinates.slice(-p.dimensions))} ×{" "}
            {coord(p.stride)} − {coord(p.before)} +{" "}
            {coord(selected.weightCoordinates.slice(2))} × {coord(p.dilation)} ={" "}
            {coord(selected.inputCoordinates.slice(-p.dimensions))}
          </span>
          <small>
            Applied separately on each spatial axis. Dilation spaces kernel
            samples apart; stride moves between output positions.
          </small>
        </div>
        <div className="linear-equation">
          <div>
            <span>Selected product · term {cursor.term}</span>
            <code>
              {selected.padding
                ? `0 (padding at ${coord(selected.inputCoordinates)})`
                : `x${coord(selected.inputCoordinates)}`}{" "}
              × W{coord(selected.weightCoordinates)}
            </code>
            {numeric && (
              <strong>
                {formatValue(xAt(selected))} ×{" "}
                {formatValue(weightData.valueAt(selected.weight))} ≈{" "}
                {formatValue(
                  finiteProduct(
                    xAt(selected),
                    weightData.valueAt(selected.weight),
                  ),
                )}
              </strong>
            )}
          </div>
          <div>
            <span>
              {complete
                ? "Sum of all products"
                : `Partial sum · terms ${window[0].term}–${window.at(-1)!.term}`}
            </span>
            <code>
              {p.channels} channels × {p.kernel.join(" × ")} kernel entries
            </code>
            {numeric && (
              <strong>
                {sum === undefined ? "—" : `≈ ${formatValue(sum)}`}
              </strong>
            )}
            {!complete && (
              <small>
                Only this {window.length}-term window is summed. Other terms
                also contribute.
              </small>
            )}
          </div>
          <div className="linear-recorded-result">
            <span>Recorded output</span>
            <code>{reference}</code>
            {numeric && (
              <strong
                title={String(
                  outputData.valueAt(cursor.output) ?? "unavailable",
                )}
              >
                {formatValue(outputData.valueAt(cursor.output))}
              </strong>
            )}
            {p.bias && (
              <small>
                Bias b[{selected.feature}]
                {numeric ? ` = ${formatValue(bias)}` : " is added once"}
              </small>
            )}
            {numeric && total !== undefined && (
              <small>
                Sum{p.bias ? " + bias" : ""} ≈ {formatValue(total)}
              </small>
            )}
          </div>
        </div>
        {numeric && (
          <p className="linear-numeric-note" role="status">
            {error
              ? "Some recorded values could not be loaded."
              : sum === undefined
                ? "A required value is missing, loading, or non-finite; no numerical sum is shown."
                : "Arithmetic uses captured values and virtual zero padding. Rounded browser calculations can differ from PyTorch accumulation; the recorded output is authoritative."}
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
            Inspect matching products{" "}
            <span>
              {window[0].term}–{window.at(-1)!.term} /{" "}
              {p.terms.toLocaleString()} terms
            </span>
          </summary>
          <div className="linear-term-navigation">
            <p>
              Terms follow local input channel, then kernel position. Dashed
              entries read zero padding.
            </p>
            <IndexControl
              label="Contribution window"
              name="Convolution contribution window"
              value={window[0].term}
              size={p.terms}
              step={8}
              onChange={selectTerm}
            />
          </div>
          <div className="linear-term-table">
            <table>
              <thead>
                <tr>
                  <th>Term</th>
                  <th>Input coordinate</th>
                  <th>Kernel coordinate</th>
                  <th>Product</th>
                </tr>
              </thead>
              <tbody>
                {window.map((s) => (
                  <tr
                    key={s.term}
                    className={`${cursor.term === s.term ? "selected" : ""} ${s.padding ? "convolution-padding" : ""}`}
                  >
                    <th scope="row">
                      <button
                        aria-label={`Inspect convolution term ${s.term}`}
                        aria-pressed={cursor.term === s.term}
                        onClick={() => selectTerm(s.term)}
                      >
                        {s.term}
                      </button>
                    </th>
                    <td
                      title={
                        numeric ? String(xAt(s) ?? "unavailable") : undefined
                      }
                    >
                      {s.padding
                        ? `Zero padding ${coord(s.inputCoordinates)}`
                        : coord(s.inputCoordinates)}
                      {numeric && !s.padding && ` = ${formatValue(xAt(s))}`}
                    </td>
                    <td
                      title={
                        numeric
                          ? String(
                              weightData.valueAt(s.weight) ?? "unavailable",
                            )
                          : undefined
                      }
                    >
                      {coord(s.weightCoordinates)}
                      {numeric &&
                        ` = ${formatValue(weightData.valueAt(s.weight))}`}
                    </td>
                    <td>
                      {numeric
                        ? formatValue(
                            finiteProduct(xAt(s), weightData.valueAt(s.weight)),
                          )
                        : "x × W"}
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

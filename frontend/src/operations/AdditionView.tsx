import { useState } from "react";
import { ArrowRight, MousePointer2 } from "lucide-react";
import type { Run } from "../api/client";
import { TensorCard } from "../tensors/TensorCard";
import { ValuesToggle } from "../tensors/ValuesToggle";
import { useTensorValues } from "../tensors/useTensorValues";
import { formatValue } from "../tensors/coordinates";
import {
  additionInputSelection,
  additionSelection,
  broadcastAxes,
  finiteAddition,
  type Addition,
} from "./addition";

const coordinate = (values: number[]) => `[${values.join(", ")}]`;

export function AdditionView({
  addition: p,
  run,
  showValues,
  onShowValues,
  onDetails,
}: {
  addition: Addition;
  run: Run;
  showValues: boolean;
  onShowValues: (show: boolean) => void;
  onDetails: () => void;
}) {
  const [output, setOutput] = useState(0);
  const selected = additionSelection(p, output);
  const shapeOnly = p.output.value_source === "shape";
  const numeric = showValues && !shapeOnly;
  const leftData = useTensorValues(
    p.left,
    run.id,
    numeric ? [selected.left.index] : [],
  );
  const rightData = useTensorValues(
    p.right,
    run.id,
    numeric ? [selected.right.index] : [],
  );
  const outputData = useTensorValues(p.output, run.id, numeric ? [output] : []);
  const datasets = [leftData, rightData, outputData];
  const error = datasets.find((d) => d.error)?.error;
  const loading = datasets.some((d) => d.loading);
  const a = leftData.valueAt(selected.left.index),
    b = rightData.valueAt(selected.right.index);
  const y = outputData.valueAt(output);
  const reconstructed = finiteAddition(a, b, p.alpha);
  const gridFrame = {
    rows: Math.max(
      ...[p.left, p.right, p.output].map((t) =>
        Math.min(8, t.shape.at(-2) ?? 1),
      ),
    ),
    columns: Math.max(
      ...[p.left, p.right, p.output].map((t) =>
        Math.min(8, t.shape.at(-1) ?? 1),
      ),
    ),
  };
  const broadcastNote = (side: "left" | "right") => {
    const axes = broadcastAxes(p[side], p.output);
    return axes.length
      ? `Reused along output ${axes.length === 1 ? "axis" : "axes"} ${axes.join(", ")}.`
      : "One matching element per output coordinate.";
  };
  return (
    <section
      className="linear-lesson addition-lesson"
      aria-label="Elementwise addition lesson"
    >
      <header className="linear-heading">
        <div>
          <code>Y = A + {p.alpha === 1 ? "B" : `${p.alpha} × B`}</code>
          <p>
            Follow one output cell back to both operands. Size-one axes reuse a
            coordinate when broadcasting.
          </p>
        </div>
        <button className="secondary-button" onClick={onDetails}>
          Tensor details <ArrowRight size={13} />
        </button>
      </header>
      <div className="linear-controls">
        <span className="addition-axis-hint">
          Axes align from the right · 0-based coordinates
        </span>
        <ValuesToggle
          checked={showValues}
          onChange={onShowValues}
          shapeOnly={shapeOnly}
        />
      </div>
      <div className="linear-tensors">
        <TensorCard
          tensor={p.left}
          label="A · Left operand"
          runId={run.id}
          gridFrame={gridFrame}
          showValues={showValues}
          focusIndex={selected.left.index}
          highlights={[selected.left.index]}
          onSelect={(index) =>
            setOutput(additionInputSelection(p, "left", index, output))
          }
        />
        <TensorCard
          tensor={p.right}
          label="B · Right operand"
          runId={run.id}
          gridFrame={gridFrame}
          showValues={showValues}
          focusIndex={selected.right.index}
          highlights={[selected.right.index]}
          onSelect={(index) =>
            setOutput(additionInputSelection(p, "right", index, output))
          }
        />
        <TensorCard
          tensor={p.output}
          label="Y · Result"
          tone="output"
          runId={run.id}
          gridFrame={gridFrame}
          showValues={showValues}
          focusIndex={output}
          highlights={[output]}
          onSelect={setOutput}
        />
      </div>
      <div className="linear-selection-note">
        <MousePointer2 size={14} />
        <p>
          Select any cell to follow its contributors. Selecting a broadcast
          operand keeps the current output batch and slice.
        </p>
      </div>
      <section className="linear-calculation" aria-label="Addition calculation">
        <header>
          <div>
            <span className="eyebrow">How this cell is formed</span>
            <code>
              A{coordinate(selected.left.coordinates)} +{" "}
              {p.alpha === 1 ? "" : `${p.alpha} × `}B
              {coordinate(selected.right.coordinates)} → Y
              {coordinate(selected.coordinates)}
            </code>
          </div>
        </header>
        <div className="linear-equation">
          <div>
            <span>Left operand</span>
            <code>A{coordinate(selected.left.coordinates)}</code>
            {numeric && (
              <strong title={String(a ?? "unavailable")}>
                {formatValue(a)}
              </strong>
            )}
            <small>{broadcastNote("left")}</small>
          </div>
          <div>
            <span>
              {p.alpha === 1
                ? "Right operand"
                : `Right operand · scaled by ${p.alpha}`}
            </span>
            <code>B{coordinate(selected.right.coordinates)}</code>
            {numeric && (
              <strong title={String(b ?? "unavailable")}>
                {formatValue(b)}
              </strong>
            )}
            <small>{broadcastNote("right")}</small>
          </div>
          <div className="linear-recorded-result">
            <span>Recorded output</span>
            <code>Y{coordinate(selected.coordinates)}</code>
            {numeric && (
              <strong title={String(y ?? "unavailable")}>
                {formatValue(y)}
              </strong>
            )}
            {numeric && reconstructed !== undefined && (
              <small>
                From the captured operands ≈ {formatValue(reconstructed)}
              </small>
            )}
          </div>
        </div>
        <p className="linear-numeric-note">
          {shapeOnly
            ? "This run records shapes and coordinates without numeric values."
            : loading && numeric
              ? "Loading the selected values…"
              : "The recorded output is authoritative. Displayed values are rounded; integer overflow and floating-point arithmetic can affect the result."}
        </p>
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
      </section>
    </section>
  );
}

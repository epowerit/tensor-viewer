import { useState } from "react";
import { ArrowRight } from "lucide-react";
import type { Run } from "../api/client";
import { TensorCard } from "../tensors/TensorCard";
import { ValuesToggle } from "../tensors/ValuesToggle";
import { useTensorValues } from "../tensors/useTensorValues";
import { formatValue } from "../tensors/coordinates";
import {
  selectScoreOperand,
  windowScoreSelection,
  type WindowScores,
} from "./windowScores";

export function WindowScoresView({
  lesson: p,
  run,
  showValues,
  onShowValues,
  onStep,
  onDetails,
}: {
  lesson: WindowScores;
  run: Run;
  showValues: boolean;
  onShowValues: (v: boolean) => void;
  onStep: (id: string) => void;
  onDetails?: () => void;
}) {
  const [selected, setSelected] = useState(0);
  const s = windowScoreSelection(p, selected);
  const shapeOnly = p.weights.value_source === "shape",
    numeric = showValues && !shapeOnly;
  const scores = useTensorValues(p.scores, run.id, numeric ? [selected] : []);
  const bias = useTensorValues(p.bias, run.id, numeric ? [s.bias] : []);
  const mask = useTensorValues(p.mask, run.id, numeric ? [s.mask] : []);
  const weights = useTensorValues(p.weights, run.id, numeric ? [selected] : []);
  const lookup = useTensorValues(p.lookup, run.id, numeric ? [s.lookup] : []);
  const datasets = [scores, bias, mask, weights, lookup],
    error = datasets.find((d) => d.error)?.error;
  const blocked = mask.valueAt(s.mask),
    index = lookup.valueAt(s.lookup);
  const value = (v: number | string | undefined) =>
    v === undefined ? (shapeOnly ? "Not recorded" : "…") : formatValue(v);
  const frame = {
    rows: Math.min(8, p.weights.shape[2]),
    columns: Math.min(8, p.weights.shape[3]),
  };
  return (
    <section
      className="window-score-lesson"
      aria-label="Window bias and mask lesson"
    >
      <header className="linear-heading">
        <div>
          <code>scores + relative bias → mask → softmax</code>
          <p>
            Bias is shared across windows and batches. The mask blocks pairs
            that crossed opposite image edges during the cyclic shift. Each
            query normalizes over its allowed keys.
          </p>
        </div>
        {onDetails && (
          <button className="secondary-button" onClick={onDetails}>
            Tensor details <ArrowRight size={13} />
          </button>
        )}
      </header>
      <div className="grouping-progression score-progression">
        {[
          [2, "Add learned bias"],
          [4, "Block wraparound"],
          [6, "Normalize each query"],
        ].map(([i, label]) => {
          const op = p.operations[Number(i)];
          return (
            <button
              className="grouping-step"
              key={op.id}
              onClick={() => onStep(op.id)}
            >
              <span>{label}</span>
              <code>[{run.trace.tensors[op.outputs[0]].shape.join(", ")}]</code>
            </button>
          );
        })}
      </div>
      <div className="grouping-selection-bar">
        <div>
          <span>
            Batch <b>{s.batch}</b>
          </span>
          <span>
            Window <b>{s.window}</b>
          </span>
          <span>
            Head <b>{s.head}</b>
          </span>
          <span>
            Query <b>{s.query}</b> → key <b>{s.key}</b>
          </span>
        </div>
        <ValuesToggle
          checked={showValues}
          onChange={onShowValues}
          shapeOnly={shapeOnly}
        />
      </div>
      <div className="window-score-tensors">
        <TensorCard
          tensor={p.bias}
          label="Relative bias"
          runId={run.id}
          gridFrame={frame}
          showValues={showValues}
          focusIndex={s.bias}
          highlights={[s.bias]}
          onSelect={(i) =>
            setSelected(selectScoreOperand(p, i, "bias", selected))
          }
        />
        <TensorCard
          tensor={{
            ...p.mask,
            axes: ["windows", "broadcast_heads", "query_tokens", "key_tokens"],
          }}
          label="Blocked pairs"
          runId={run.id}
          gridFrame={frame}
          showValues={showValues}
          focusIndex={s.mask}
          highlights={[s.mask]}
          onSelect={(i) =>
            setSelected(selectScoreOperand(p, i, "mask", selected))
          }
        />
        <TensorCard
          tensor={p.weights}
          label="Attention weights"
          tone="output"
          runId={run.id}
          gridFrame={frame}
          showValues={showValues}
          focusIndex={selected}
          highlights={[selected]}
          onSelect={setSelected}
        />
      </div>
      <section
        className="grouping-coordinate-path"
        aria-label="Selected attention pair"
      >
        <div>
          <span>Query position in window</span>
          <code>[{s.queryPosition.join(", ")}]</code>
        </div>
        <ArrowRight size={16} />
        <div>
          <span>Key position in window</span>
          <code>[{s.keyPosition.join(", ")}]</code>
        </div>
        <p>
          Relative offset (query − key): [
          {s.queryPosition.map((v, i) => v - s.keyPosition[i]).join(", ")}].{" "}
          {numeric && index !== undefined
            ? `Recorded bias lookup: table[${index}, ${s.head}].`
            : "The learned table has one entry per relative offset and head."}
        </p>
        <small>
          Mask 1 = blocked; mask 0 = allowed. The mask is reused across image
          batches and heads.
        </small>
      </section>
      {numeric && (
        <section
          className="window-score-calculation"
          aria-label="Recorded attention calculation"
        >
          <span>
            Scaled dot product <b>{value(scores.valueAt(selected))}</b>
          </span>
          <span>
            Relative bias <b>{value(bias.valueAt(s.bias))}</b>
          </span>
          <span>
            {blocked === undefined
              ? "Mask loading"
              : Number(blocked)
                ? "Blocked → −∞ before softmax"
                : "Allowed key"}
          </span>
          <span>
            Recorded weight <b>{value(weights.valueAt(selected))}</b>
          </span>
          <p>
            Softmax uses the entire key row. A blocked pair has zero weight;
            allowed weights depend on all scores in that row.
          </p>
        </section>
      )}
      {error && (
        <div role="alert" className="operation-error">
          <p>{error}</p>
          <button
            className="secondary-button"
            onClick={() => datasets.forEach((d) => d.retry())}
          >
            Retry values
          </button>
        </div>
      )}
    </section>
  );
}

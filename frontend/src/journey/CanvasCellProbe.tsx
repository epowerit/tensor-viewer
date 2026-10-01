import { Maximize2, X } from "lucide-react";
import type { Tensor } from "../api/client";
import { formatValue, unravel } from "../tensors/coordinates";
import { useTensorValues } from "../tensors/useTensorValues";
import type { CellContributors } from "./cellContributors";
import "./cellProbe.css";

export type CanvasProbe = {
  tensor: Tensor;
  index: number;
  result: CellContributors;
  runId: string;
};

/** A selected cell stays on the canvas; values come from its immutable snapshot. */
export function CanvasCellProbe({
  probe,
  onInspect,
  onClear,
}: {
  probe: CanvasProbe;
  onInspect: () => void;
  onClear: () => void;
}) {
  const { tensor, index, result, runId } = probe;
  const values = useTensorValues(tensor, runId, [index]);
  const value =
    tensor.value_source === "shape" ? undefined : values.valueAt(index);
  const coordinate = `[${unravel(index, tensor.shape).join(", ")}]`;
  return (
    <div
      className="canvas-cell-probe"
      role="region"
      aria-label="Selected cell contributors"
    >
      <div className="cell-probe-heading">
        <strong title={`${tensor.name}${coordinate}`}>
          {tensor.name}
          <span>{coordinate}</span>
        </strong>
        <code title={value === undefined ? undefined : String(value)}>
          {values.loading
            ? "…"
            : tensor.value_source === "shape"
              ? "shape only"
              : value === undefined
                ? "value unavailable"
                : `= ${formatValue(value)}`}
        </code>
        <button
          aria-label="Enlarge selected cell"
          title="Inspect this cell in 3D"
          onClick={onInspect}
        >
          <Maximize2 size={14} />
        </button>
        <button
          aria-label="Clear cell tracing"
          title="Return to the operation scene (Escape)"
          onClick={onClear}
        >
          <X size={14} />
        </button>
      </div>
      <p role="status" title={result.summary}>
        {result.truncated && (
          <b>
            {result.sources.length} of {result.total.toLocaleString()} source
            uses tracked ·{" "}
          </b>
        )}
        {result.summary}
      </p>
      {values.error && (
        <button className="cell-probe-retry" onClick={values.retry}>
          Retry loading value
        </button>
      )}
    </div>
  );
}

import { useContext } from "react";
import type { Tensor } from "../api/client";
import { ContractContext } from "./ContractContext";
import { heatFill, heatLevel, heatRangeOf, useHeat } from "../tensors/heat";
import { ValueSpread } from "../tensors/ValueSpread";
import { pixelColors, pixelPlan } from "../inputs/samples";
import { InkSize, useAxisInk, useCellPaint } from "../tensors/InkShape";
import { formatCellValue, formatValue } from "../tensors/coordinates";
import {
  describeAxis,
  explainedLineage,
  type AxisStory,
} from "../tensors/axisLineage";

/** A hover card: what a tensor is, and a first look at its recorded values. */
export function TensorPeek({
  tensor,
  lineage,
}: {
  tensor: Tensor;
  /** Where each axis comes from, when it was traced. */
  lineage?: AxisStory[] | null;
}) {
  const explained = explainedLineage(tensor, lineage);
  const ink = useAxisInk(tensor);
  const paint = useCellPaint();
  const rank = tensor.shape.length;
  const inline = tensor.values.length === tensor.numel && tensor.numel > 0;
  const rows = Math.min(rank > 1 ? tensor.shape[rank - 2] : 1, 6),
    columns = Math.min(rank ? tensor.shape[rank - 1] : 1, 8);
  const width = rank ? tensor.shape[rank - 1] : 1;
  const magnitude = Math.max(
    Math.abs(tensor.minimum ?? 0),
    Math.abs(tensor.maximum ?? 0),
    1e-12,
  );
  const plan = inline ? pixelPlan(tensor, 0) : null;
  // With Heat on, the first window is shaded as every grid is.
  const [heat] = useHeat();
  const heatRange = heat ? heatRangeOf(tensor) : null;
  const contract = useContext(ContractContext)?.byTensor.get(tensor.id);
  const picture = plan ? pixelColors(plan, (i) => tensor.values[i]) : null;
  return (
    <div className="tensor-peek" role="tooltip">
      <header>
        <b>{tensor.name}</b>
        <span>{tensor.dtype}</span>
      </header>
      <div className="peek-shape">
        {rank ? (
          tensor.shape.map((size, axis) => (
            <span key={axis}>
              <i>{tensor.axes[axis] ?? `axis ${axis}`}</i>
              <InkSize size={size} ink={ink?.[axis]} />
            </span>
          ))
        ) : (
          <span>
            <i>scalar</i>1
          </span>
        )}
      </div>
      {picture && plan ? (
        <svg
          className="peek-picture"
          viewBox={`0 0 ${plan.width} ${plan.height}`}
          shapeRendering="crispEdges"
          aria-hidden="true"
        >
          {picture.colors.map((color, p) => (
            <rect
              key={p}
              x={p % plan.width}
              y={Math.floor(p / plan.width)}
              width={1}
              height={1}
              fill={color ?? "transparent"}
            />
          ))}
        </svg>
      ) : inline ? (
        <div
          className="peek-grid"
          style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}
          aria-hidden="true"
        >
          {Array.from({ length: rows * columns }, (_, i) => {
            const flat = Math.floor(i / columns) * width + (i % columns);
            const value = tensor.values[flat];
            // Glass tinted by where the value came from, as in the views.
            const level =
              heatRange && typeof value === "number"
                ? heatLevel(value, heatRange.low, heatRange.high)
                : null;
            const tint = level === null ? paint?.(tensor, flat) : null;
            const strength =
              typeof value === "number" && Number.isFinite(value)
                ? Math.abs(value) / magnitude
                : 0;
            return (
              <span
                key={i}
                className={tint ? "peek-inked" : undefined}
                style={
                  level !== null
                    ? { background: heatFill(level) }
                    : tint
                      ? ({ "--cell-ink": tint } as React.CSSProperties)
                      : {
                          background: `rgb(167 139 250 / ${(0.08 + 0.62 * strength).toFixed(2)})`,
                        }
                }
              >
                {formatCellValue(value, 4)}
              </span>
            );
          })}
        </div>
      ) : (
        <p>
          {tensor.value_source === "shape"
            ? "Shape-only run: no values were recorded."
            : "Open the step to page through its values."}
        </p>
      )}
      {tensor.value_source !== "shape" && tensor.histogram && (
        <ValueSpread tensor={tensor} />
      )}
      {contract && (
        <p
          className={`peek-contract ${contract.ok ? "peek-contract-kept" : "peek-contract-broken"}`}
        >
          {contract.ok ? "✓" : "✗"} {contract.text}
          {!contract.ok && <small>{contract.message}</small>}
        </p>
      )}
      {explained && (
        <div className="peek-lineage">
          <span>Axes come from</span>
          {explained.map((story, axis) => (
            <p key={axis}>
              <b>{story.size}</b>
              {describeAxis(story)}
            </p>
          ))}
        </div>
      )}
      <footer>
        {tensor.numel.toLocaleString()} elements
        {tensor.minimum != null &&
          tensor.maximum != null &&
          ` · ${formatValue(tensor.minimum)} … ${formatValue(tensor.maximum)}`}
        {inline && (rows * columns < tensor.numel ? " · first window" : "")}
        <br />
        {tensor.storage_id}
        {tensor.contiguous ? "" : " · strided view"}
      </footer>
    </div>
  );
}

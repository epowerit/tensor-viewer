import type { Tensor } from "../api/client";
import { pixelColors, pixelPlan } from "../inputs/samples";
import { formatCellValue, formatValue } from "../tensors/coordinates";
import {
  describeAxis,
  isOwnLineage,
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
  const explained =
    lineage &&
    lineage.length === tensor.shape.length &&
    !isOwnLineage(tensor, lineage)
      ? lineage
      : null;
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
              {size}
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
            const value =
              tensor.values[Math.floor(i / columns) * width + (i % columns)];
            const strength =
              typeof value === "number" && Number.isFinite(value)
                ? Math.abs(value) / magnitude
                : 0;
            return (
              <span
                key={i}
                style={{
                  background: `rgb(167 139 250 / ${(0.08 + 0.62 * strength).toFixed(2)})`,
                }}
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

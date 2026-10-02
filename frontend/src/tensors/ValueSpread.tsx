import type { Tensor } from "../api/client";
import "./valueSpread.css";

const format = (value: number) =>
  Math.abs(value) >= 1000 || (value !== 0 && Math.abs(value) < 0.01)
    ? value.toExponential(1)
    : value.toFixed(2);

/**
 * How a tensor's values are spread: a histogram of its finite values, the
 * zero line, and the chosen cell's value marked in fire. Dead activations (a
 * spike at zero), saturation, outliers and NaN or infinity show at a glance.
 */
export function ValueSpread({
  tensor,
  value,
}: {
  tensor: Tensor;
  /** The chosen or hovered cell's value, marked on the histogram. */
  value?: number | string;
}) {
  const histogram = tensor.histogram;
  if (!histogram || !histogram.counts.length) {
    return histogram?.non_finite ? (
      <p className="value-spread value-spread-broken">
        All {histogram.non_finite.toLocaleString()} values are NaN or infinite.
      </p>
    ) : null;
  }
  const { low, high, counts, zeros, non_finite: broken } = histogram;
  const width = 176,
    height = 26;
  const tallest = Math.max(...counts, 1);
  const step = width / counts.length;
  const x = (point: number) =>
    high === low ? width / 2 : ((point - low) / (high - low)) * width;
  const marked =
    typeof value === "number" && Number.isFinite(value) ? value : null;
  const total = counts.reduce((sum, count) => sum + count, 0) + broken;
  const zeroShare = total ? (zeros / total) * 100 : 0;
  const summary = [
    histogram.mean !== null && histogram.mean !== undefined
      ? `mean ${format(histogram.mean)}`
      : null,
    histogram.std !== null && histogram.std !== undefined
      ? `σ ${format(histogram.std)}`
      : null,
    zeros ? `${zeroShare < 1 ? "<1" : Math.round(zeroShare)}% zeros` : null,
  ].filter(Boolean);
  return (
    <div className="value-spread">
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${tensor.name} values from ${format(low)} to ${format(high)}, ${summary.join(", ")}${broken ? `, ${broken} NaN or infinite` : ""}${marked !== null ? `; the chosen value ${format(marked)} is marked` : ""}`}
      >
        {counts.map((count, i) => {
          const barHeight = count
            ? Math.max(1.5, (count / tallest) * height)
            : 0;
          return (
            <rect
              key={i}
              x={i * step + 0.5}
              y={height - barHeight}
              width={Math.max(1, step - 1)}
              height={barHeight}
              rx="1"
            />
          );
        })}
        {low < 0 && high > 0 && (
          <line
            className="value-spread-zero"
            x1={x(0)}
            x2={x(0)}
            y1={0}
            y2={height}
          />
        )}
        {marked !== null && (
          <line
            className="value-spread-mark"
            x1={x(marked)}
            x2={x(marked)}
            y1={0}
            y2={height}
          />
        )}
      </svg>
      <span className="value-spread-axis">
        <span>{format(low)}</span>
        <span>{summary.join(" · ")}</span>
        <span>{format(high)}</span>
      </span>
      {broken > 0 && (
        <span className="value-spread-broken">
          {broken.toLocaleString()} NaN or infinite
        </span>
      )}
    </div>
  );
}

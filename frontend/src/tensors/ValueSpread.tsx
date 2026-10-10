import { useState } from "react";
import type { Histogram, Tensor } from "../api/client";
import "./valueSpread.css";
import { formatCount } from "../formatCount";

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
  onFind,
  bin = null,
  onBin,
  before = null,
}: {
  tensor: Tensor;
  /** The chosen or hovered cell's value, marked on the histogram. */
  value?: number | string;
  /**
   * When given, the range ends and the NaN count become buttons that select
   * the smallest value, the largest, or the next non-finite one.
   */
  onFind?: { min?: () => void; max?: () => void; broken?: () => void };
  /** The bar chosen to ring its cells, if any. */
  bin?: number | null;
  /** When given, each bar is a button choosing (or clearing) its bin. */
  onBin?: (bin: number) => void;
  /** The same tensor's histogram in the run before, drawn as an outline. */
  before?: Histogram | null;
}) {
  const histogram = tensor.histogram;
  // Log counts keep a spike, such as ReLU's zeros, from flattening the rest.
  const [log, setLog] = useState(false);
  const end = (text: string, find: (() => void) | undefined, what: string) =>
    find ? (
      <button
        type="button"
        className="value-spread-find"
        title={`Select the ${what} value`}
        aria-label={`Select the ${what} value, ${text}`}
        onClick={find}
      >
        {text}
      </button>
    ) : (
      <span>{text}</span>
    );
  const brokenNote = (text: string) =>
    onFind?.broken ? (
      <button
        type="button"
        className="value-spread-broken value-spread-find"
        title="Select the next NaN or infinite value"
        onClick={onFind.broken}
      >
        {text}
      </button>
    ) : (
      <span className="value-spread-broken">{text}</span>
    );
  if (!histogram || !histogram.counts.length) {
    return histogram?.non_finite ? (
      <p className="value-spread value-spread-broken">
        {brokenNote(
          `All ${formatCount(histogram.non_finite)} values are NaN or infinite.`,
        )}
      </p>
    ) : null;
  }
  const { low, high, counts, zeros, non_finite: broken } = histogram;
  const width = 176,
    height = 26;
  const earlier = before?.counts.length ? before : null;
  // One height scale for both, so a bar that grew reads as taller.
  const scale = (count: number) => (log ? Math.log1p(count) : count);
  const tallest = scale(Math.max(...counts, ...(earlier?.counts ?? []), 1));
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
        role={onBin ? "group" : "img"}
        aria-label={`${tensor.name} values from ${format(low)} to ${format(high)}, ${summary.join(", ")}${broken ? `, ${broken} NaN or infinite` : ""}${marked !== null ? `; the chosen value ${format(marked)} is marked` : ""}`}
      >
        {counts.map((count, i) => {
          const barHeight = count
            ? Math.max(1.5, (scale(count) / tallest) * height)
            : 0;
          return (
            <rect
              key={i}
              className={i === bin ? "value-spread-chosen" : undefined}
              x={i * step + 0.5}
              y={height - barHeight}
              width={Math.max(1, step - 1)}
              height={barHeight}
              rx="1"
            />
          );
        })}
        {onBin &&
          counts.map((count, i) => {
            const from = low + ((high - low) * i) / counts.length;
            const to = low + ((high - low) * (i + 1)) / counts.length;
            return (
              <rect
                key={`hit-${i}`}
                className="value-spread-hit"
                x={i * step}
                y={0}
                width={step}
                height={height}
                role="button"
                tabIndex={count ? 0 : -1}
                aria-pressed={i === bin}
                aria-label={`${formatCount(count)} values from ${format(from)} to ${format(to)}; ring them in the grid`}
                onClick={() => count && onBin(i)}
                onKeyDown={(event) => {
                  if (count && (event.key === "Enter" || event.key === " ")) {
                    event.preventDefault();
                    onBin(i);
                  }
                }}
              >
                <title>{`${formatCount(count)} values from ${format(from)} to ${format(to)}`}</title>
              </rect>
            );
          })}
        {earlier && (
          <path
            className="value-spread-before"
            d={beforeOutline(earlier, x, width, height, tallest, scale)}
          >
            <title>The run before</title>
          </path>
        )}
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
      <button
        type="button"
        className="value-spread-scale"
        aria-pressed={log}
        title={
          log
            ? "Bar heights are log counts. Switch back to plain counts."
            : "Show log counts, so a spike (such as ReLU's zeros) does not flatten the rest"
        }
        onClick={() => setLog(!log)}
      >
        log
      </button>
      <span className="value-spread-axis">
        {end(format(low), onFind?.min, "smallest")}
        <span>{summary.join(" · ")}</span>
        {end(format(high), onFind?.max, "largest")}
      </span>
      {broken > 0 && brokenNote(`${formatCount(broken)} NaN or infinite`)}
    </div>
  );
}

/**
 * An earlier histogram as a stepped outline on the current value scale; its
 * bars sit where their values fall, clipped to the drawing.
 */
function beforeOutline(
  earlier: Histogram,
  x: (value: number) => number,
  width: number,
  height: number,
  tallest: number,
  scale: (count: number) => number,
) {
  const step = (earlier.high - earlier.low) / earlier.counts.length;
  const clamp = (value: number) => Math.max(0, Math.min(width, value));
  const parts: string[] = [];
  earlier.counts.forEach((count, i) => {
    const left = clamp(
      earlier.high === earlier.low
        ? x(earlier.low) - 1
        : x(earlier.low + i * step),
    );
    const right = clamp(
      earlier.high === earlier.low
        ? x(earlier.low) + 1
        : x(earlier.low + (i + 1) * step),
    );
    const top = height - (scale(count) / tallest) * height;
    parts.push(
      `${i ? "L" : "M"}${left.toFixed(1)},${(i ? top : height).toFixed(1)}`,
    );
    if (!i) parts.push(`L${left.toFixed(1)},${top.toFixed(1)}`);
    parts.push(`L${right.toFixed(1)},${top.toFixed(1)}`);
  });
  const last = clamp(x(earlier.high));
  parts.push(`L${last.toFixed(1)},${height}`);
  return parts.join(" ");
}

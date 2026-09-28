import { useEffect, useState } from "react";
import { Layers3 } from "lucide-react";
import type { Tensor } from "../api/client";
import { formatValue, product, ravel, unravel } from "./coordinates";

type Props = {
  tensor: Tensor;
  label: string;
  tone?: "input" | "output";
  highlights?: number[];
  focusIndex?: number;
  showValues: boolean;
  onSelect?: (index: number) => void;
};

function axisColor(name: string, index: number) {
  if (/batch/.test(name)) return 0;
  if (/features|channels/.test(name)) return 3;
  if (/heads|groups/.test(name)) return 2;
  if (/tokens/.test(name)) return 1;
  return index % 4;
}

export function TensorCard({
  tensor,
  label,
  tone = "input",
  highlights = [],
  focusIndex,
  showValues,
  onSelect,
}: Props) {
  const rank = tensor.shape.length;
  const leading = Math.max(0, rank - 2);
  const [slice, setSlice] = useState<number[]>(Array(leading).fill(0));
  const [windowStart, setWindowStart] = useState([0, 0]);
  useEffect(() => {
    const coords = tensor.numel
      ? unravel(focusIndex ?? 0, tensor.shape)
      : Array(rank).fill(0);
    setSlice(coords.slice(0, leading));
    setWindowStart([
      Math.floor((coords.at(-2) ?? 0) / 8) * 8,
      Math.floor((coords.at(-1) ?? 0) / 8) * 8,
    ]);
  }, [tensor.id, focusIndex, leading, tensor.shape, tensor.numel, rank]);
  const rows = rank < 2 ? 1 : tensor.shape[rank - 2];
  const cols = rank === 0 ? 1 : tensor.shape[rank - 1];
  const rowStart = Math.min(windowStart[0], Math.max(0, rows - 1));
  const colStart = Math.min(windowStart[1], Math.max(0, cols - 1));
  const visibleRows = Math.min(8, rows - rowStart);
  const visibleCols = Math.min(8, cols - colStart);
  const cell = Math.min(38, 232 / Math.max(visibleCols, 1));
  const width = visibleCols * cell;
  const height = visibleRows * 34;
  const startX = (300 - width) / 2 - 7;
  const startY = Math.max(40, (225 - height) / 2);
  const stacks = product(tensor.shape.slice(0, leading));
  const selected = new Set(highlights);
  const numberOfSheets = Math.min(3, Math.max(1, stacks));
  const maxMagnitude = Math.max(
    Math.abs(tensor.minimum ?? 0),
    Math.abs(tensor.maximum ?? 0),
    0.001,
  );
  return (
    <article className={`tensor-card tensor-${tone}`}>
      <div className="tensor-card-header">
        <span className="eyebrow">{label}</span>
        <span className="mono tensor-name">{tensor.name}</span>
      </div>
      <div
        className="shape-badges"
        aria-label={`${label} shape ${tensor.shape.join(", ")}`}
      >
        {rank === 0 ? (
          <span className="axis-badge">scalar</span>
        ) : (
          tensor.shape.map((size, i) => (
            <span
              className={`axis-badge axis-${axisColor(tensor.axes[i], i)}`}
              key={i}
              title={tensor.axes[i]}
            >
              <b>{size}</b>
              <small>{tensor.axes[i]}</small>
            </span>
          ))
        )}
      </div>
      <svg
        className="tensor-svg"
        viewBox={`0 0 300 ${Math.max(265, height + 85)}`}
        aria-label={`${tensor.name} tensor, ${tensor.shape.join(" by ")}`}
      >
        {tensor.numel === 0 && (
          <text x="150" y="120" textAnchor="middle" className="svg-axis">
            Empty tensor · no elements
          </text>
        )}
        {Array.from(
          { length: numberOfSheets - 1 },
          (_, i) => numberOfSheets - 1 - i,
        ).map((i) => (
          <rect
            key={i}
            x={startX + i * 8}
            y={startY - i * 9}
            width={width}
            height={height}
            rx="5"
            fill={tone === "output" ? "#e5f3ed" : "#ecedf5"}
            stroke={tone === "output" ? "#9bc8bb" : "#b8bfd4"}
          />
        ))}
        {Array.from({ length: visibleRows }, (_, row) =>
          Array.from({ length: visibleCols }, (_, col) => {
            const coords =
              rank === 0
                ? []
                : rank === 1
                  ? [col + colStart]
                  : [...slice, row + rowStart, col + colStart];
            const index = ravel(coords, tensor.shape);
            const active = selected.has(index);
            const value = tensor.values[index];
            const intensity =
              showValues && typeof value === "number"
                ? Math.min(0.55, (Math.abs(value) / maxMagnitude) * 0.5)
                : 0;
            const fill = active
              ? tone === "output"
                ? "#17816c"
                : "#6875aa"
              : showValues
                ? `rgba(${tone === "output" ? "34,139,111" : "100,115,169"}, ${0.06 + intensity})`
                : tone === "output"
                  ? "#edf7f2"
                  : "#f1f2f8";
            return (
              <g
                key={index}
                role={onSelect ? "button" : undefined}
                tabIndex={onSelect ? 0 : undefined}
                aria-label={`${label} element ${coords.join(",")} value ${formatValue(value)}`}
                onClick={() => onSelect?.(index)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelect?.(index);
                  }
                }}
                className={onSelect ? "tensor-cell" : ""}
              >
                <title>{`[${coords.join(", ")}] = ${formatValue(value)}`}</title>
                <rect
                  x={startX + col * cell}
                  y={startY + row * 34}
                  width={cell}
                  height={34}
                  fill={fill}
                  stroke={
                    active ? "#fff" : tone === "output" ? "#b9d9ce" : "#cbd0e2"
                  }
                  strokeWidth={active ? 1.5 : 0.8}
                />
                {(showValues || active) && (
                  <text
                    x={startX + col * cell + cell / 2}
                    y={startY + row * 34 + 20}
                    textAnchor="middle"
                    fontSize={Math.min(
                      11,
                      (cell - 6) / (formatValue(value).length * 0.62),
                    )}
                    fill={active ? "#fff" : "#40515c"}
                  >
                    {formatValue(value)}
                  </text>
                )}
              </g>
            );
          }),
        )}
        <text
          x={startX + width / 2}
          y={startY + height + 24}
          textAnchor="middle"
          className="svg-axis"
        >
          {rank > 0 ? `${tensor.axes[rank - 1]} · ${cols}` : "one value"}
        </text>
        {rank > 1 && (
          <text
            transform={`translate(${startX - 13},${startY + height / 2}) rotate(-90)`}
            textAnchor="middle"
            className="svg-axis"
          >
            {tensor.axes[rank - 2]} · {rows}
          </text>
        )}
      </svg>
      {(rows > 8 || cols > 8) && (
        <div className="matrix-window">
          <span>
            Showing {rowStart + 1}–{rowStart + visibleRows} × {colStart + 1}–
            {colStart + visibleCols}
          </span>
          <button
            onClick={() =>
              setWindowStart([rowStart + 8 < rows ? rowStart + 8 : 0, colStart])
            }
          >
            Rows ↻
          </button>
          <button
            onClick={() =>
              setWindowStart([rowStart, colStart + 8 < cols ? colStart + 8 : 0])
            }
          >
            Columns ↻
          </button>
        </div>
      )}
      {leading > 0 && (
        <div className="slice-controls">
          {tensor.shape.slice(0, leading).map((size, i) => (
            <label key={i}>
              {tensor.axes[i]}
              <select
                aria-label={`${label} ${tensor.axes[i]} slice`}
                value={slice[i] ?? 0}
                onChange={(e) =>
                  setSlice((prev) =>
                    prev.map((v, j) => (j === i ? Number(e.target.value) : v)),
                  )
                }
              >
                {Array.from({ length: size }, (_, n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
      )}
      <div className="tensor-foot">
        <span>
          <Layers3 size={12} /> {tensor.numel.toLocaleString()} elements
        </span>
        <span>{tensor.dtype}</span>
      </div>
      <details className="memory-details">
        <summary>Storage & strides</summary>
        <dl>
          <dt>Storage</dt>
          <dd>{tensor.storage_id}</dd>
          <dt>Strides</dt>
          <dd>[{tensor.strides.join(", ")}]</dd>
          <dt>Offset</dt>
          <dd>{tensor.storage_offset}</dd>
          <dt>Contiguous</dt>
          <dd>{tensor.contiguous ? "Yes" : "No"}</dd>
        </dl>
      </details>
    </article>
  );
}

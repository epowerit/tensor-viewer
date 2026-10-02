import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { Box, Image, SlidersHorizontal } from "lucide-react";
import { pixelPlan } from "../inputs/samples";
import { PixelView } from "./PixelView";
import { describeAxis, shortAxis } from "./axisLineage";
import { useAxisOrigins } from "./LineageContext";
import { inkStyle, useAxisInk, useCellPaint } from "./InkShape";
import "./gridInk.css";
import type { Tensor } from "../api/client";
import {
  exactValue,
  formatCellValue,
  formatValue,
  product,
  ravel,
} from "./coordinates";
import {
  changePlane,
  coordinatesFor,
  defaultPlane,
  hiddenAxes,
  moveInPlane,
  planeCoordinates,
  planeSize,
  safeIndex,
  storagePosition,
} from "./plane";
import { CoordinateJump, IndexControl } from "./TensorNavigation";
import { useTensorValues } from "./useTensorValues";
import { TensorVolumeDialog } from "./TensorVolumeDialog";
import { ValueSpread } from "./ValueSpread";

type Props = {
  tensor: Tensor;
  label: string;
  runId?: string;
  tone?: "input" | "output";
  /**
   * An active tensor (a step's result, or the one being inspected) burns at
   * its chosen cell. A lit tensor already held values: its chosen cell is a
   * source, marked but not on fire. Results are active unless told otherwise.
   */
  light?: "active" | "lit";
  highlights?: number[];
  focusIndex?: number;
  showValues: boolean;
  gridFrame: { rows: number; columns: number };
  expandDetails?: boolean;
  onSelect?: (index: number) => void;
};
export function TensorCard(props: Props) {
  return <TensorExplorer key={props.tensor.id} {...props} />;
}

function TensorExplorer({
  tensor,
  label,
  runId,
  tone = "input",
  light = tone === "output" ? "active" : "lit",
  highlights = [],
  focusIndex,
  showValues,
  gridFrame,
  expandDetails = false,
  onSelect,
}: Props) {
  const id = useId();
  const rank = tensor.shape.length;
  const [plane, setPlane] = useState(() => defaultPlane(rank));
  const [cursor, setCursor] = useState(focusIndex ?? 0);
  const [hovered, setHovered] = useState<number | null>(null);
  const [settings, setSettings] = useState(false);
  const [volume, setVolume] = useState(false);
  const [pixels, setPixels] = useState(false);
  const [pageSize, setPageSize] = useState(8);
  const svg = useRef<SVGSVGElement>(null);
  const keyboardFocus = useRef(false);
  const index = safeIndex(cursor, tensor.shape);
  const coords = coordinatesFor(index, tensor.shape);
  const { rows, columns } = planeSize(tensor.shape, plane);
  const slices = hiddenAxes(tensor.shape, plane);
  const stacks = product(slices.map((axis) => tensor.shape[axis]));
  const rowStart =
    Math.floor((plane.row === null ? 0 : coords[plane.row]) / pageSize) *
    pageSize;
  const colStart =
    Math.floor((plane.column === null ? 0 : coords[plane.column]) / pageSize) *
    pageSize;
  const visibleRows = tensor.numel ? Math.min(pageSize, rows - rowStart) : 0;
  const visibleCols = tensor.numel ? Math.min(pageSize, columns - colStart) : 0;
  const frameRows = Math.max(
    1,
    visibleRows,
    Math.min(pageSize, gridFrame.rows),
  );
  const frameCols = Math.max(
    1,
    visibleCols,
    Math.min(pageSize, gridFrame.columns),
  );
  const cellW = 48,
    cellH = 30,
    startX = 56,
    startY = 30;
  const canvasWidth = frameCols * cellW + 94,
    viewHeight = Math.max(110, frameRows * cellH + 70);
  const width = visibleCols * cellW,
    height = visibleRows * cellH;
  const cells = Array.from({ length: visibleRows * visibleCols }, (_, i) => {
    const row = Math.floor(i / visibleCols),
      column = i % visibleCols;
    const coordinates = planeCoordinates(
      tensor.shape,
      plane,
      coords,
      row + rowStart,
      column + colStart,
    );
    return { row, column, coordinates, flat: ravel(coordinates, tensor.shape) };
  });
  const data = useTensorValues(
    tensor,
    runId,
    cells.map((cell) => cell.flat),
  );
  const linked = new Set(highlights);
  const readIndex = hovered ?? index,
    readCoords = coordinatesFor(readIndex, tensor.shape);
  const axisName = (axis: number) => tensor.axes[axis] || `axis ${axis}`;
  const origins = useAxisOrigins(tensor);
  const ink = useAxisInk(tensor);
  // Squares are glass tinted by where each value came from, like the cubes.
  const paint = useCellPaint();
  const shapeOnly = tensor.value_source === "shape";
  const picture = showValues && !shapeOnly ? pixelPlan(tensor, index) : null;
  const numericValues = cells
    .map((cell) => data.valueAt(cell.flat))
    .filter(
      (value): value is number =>
        typeof value === "number" && Number.isFinite(value),
    );
  const magnitude = Math.max(
    Math.abs(tensor.minimum ?? 0),
    Math.abs(tensor.maximum ?? 0),
    ...numericValues.map(Math.abs),
    0.001,
  );
  useEffect(() => {
    setCursor(focusIndex ?? 0);
    setHovered(null);
  }, [focusIndex]);
  useLayoutEffect(() => {
    if (keyboardFocus.current) {
      svg.current
        ?.querySelector<SVGGElement>(`[data-cell-index="${index}"]`)
        ?.focus({ preventScroll: true });
      keyboardFocus.current = false;
    }
  }, [index, plane, pageSize]);
  function select(next: number) {
    if (!tensor.numel) return;
    const value = safeIndex(next, tensor.shape);
    setCursor(value);
    setHovered(null);
    onSelect?.(value);
  }
  function sliceAt(axis: number, value: number) {
    select(
      ravel(
        coords.map((coordinate, i) => (i === axis ? value : coordinate)),
        tensor.shape,
      ),
    );
  }
  function jumpTo(next: number) {
    select(next);
    requestAnimationFrame(() =>
      svg.current
        ?.querySelector<SVGGElement>(`[data-cell-index="${next}"]`)
        ?.focus(),
    );
  }
  return (
    <article
      className={`tensor-card tensor-${tone}`}
      aria-label={`${label} tensor explorer`}
    >
      <header className="tensor-card-header">
        <div className="tensor-heading">
          <span className="eyebrow">{label}</span>
          <span className="tensor-name" title={tensor.name}>
            {tensor.name}
          </span>
        </div>
        <div className="tensor-card-tools">
          {picture && (
            <button
              className="tensor-view-toggle"
              aria-label={`Draw ${label} tensor as pixels`}
              aria-pressed={pixels}
              onClick={() => setPixels(!pixels)}
              title="Draw the selected height × width plane as a picture"
            >
              <Image size={14} />
              <span>Pixels</span>
            </button>
          )}
          <button
            className="tensor-view-toggle"
            aria-label={`Enlarge ${label} tensor in 3D`}
            onClick={() => setVolume(true)}
          >
            <Box size={14} />
            <span>3D</span>
          </button>
          {(rank > 1 || rows > 8 || columns > 8) && (
            <button
              className="tensor-view-toggle"
              aria-label={`${label} view settings`}
              aria-expanded={settings}
              aria-controls={`${id}-controls`}
              onClick={() => setSettings(!settings)}
              title="View settings: axes and window size"
            >
              <SlidersHorizontal size={14} /> <span>View</span>
            </button>
          )}
        </div>
      </header>
      {pixels && picture && (
        <PixelView
          tensor={tensor}
          plan={picture}
          runId={runId}
          selected={index}
          onSelect={select}
        />
      )}
      {volume && (
        <TensorVolumeDialog
          tensor={tensor}
          runId={runId}
          initialIndex={index}
          onSelect={select}
          onClose={() => setVolume(false)}
        />
      )}
      <div
        className="shape-badges"
        aria-label={`${label} shape ${tensor.shape.join(", ") || "scalar"}`}
      >
        {rank === 0 ? (
          <span className="axis-badge">scalar</span>
        ) : (
          tensor.shape.map((size, axis) => {
            const origin = origins?.[axis];
            const short = origin ? shortAxis(origin) : null;
            return (
              <span
                className={`axis-badge ${axis === plane.row || axis === plane.column ? "axis-visible" : "axis-sliced"}${ink?.[axis] ? ` inked${ink[axis]!.piece ? " ink-piece-badge" : ""}${ink[axis]!.colors.length > 1 ? " ink-merged-badge" : ""}` : ""}`}
                style={inkStyle(ink?.[axis])}
                key={axis}
                title={`Axis ${axis}: ${axisName(axis)} · ${size.toLocaleString()} · ${axis === plane.row ? "rows" : axis === plane.column ? "columns" : "slice"}${origin ? `\nFrom ${describeAxis(origin)}` : ""}`}
              >
                <small>{axisName(axis)}</small>
                <b>{size.toLocaleString()}</b>
                {short &&
                  !(
                    origin!.terms.length === 1 &&
                    !origin!.terms[0].part &&
                    origin!.terms[0].label ===
                      `${tensor.name}.${axisName(axis)}`
                  ) && <i className="axis-origin">← {short}</i>}
              </span>
            );
          })
        )}
      </div>
      {slices.some((axis) => tensor.shape[axis] > 1) && !!tensor.numel && (
        <div className="tensor-slice-bar">
          {slices
            .filter((axis) => tensor.shape[axis] > 1)
            .map((axis) => (
              <IndexControl
                key={axis}
                label={axisName(axis)}
                name={`${label} ${axisName(axis)} slice`}
                value={coords[axis]}
                size={tensor.shape[axis]}
                onChange={(value) => sliceAt(axis, value)}
              />
            ))}
        </div>
      )}
      <div className="tensor-drawing">
        <svg
          ref={svg}
          className="tensor-svg"
          viewBox={`0 0 ${canvasWidth} ${viewHeight}`}
          role="group"
          aria-label={`${tensor.name} tensor, ${tensor.shape.join(" by ") || "scalar"}`}
          onMouseLeave={() => setHovered(null)}
        >
          <title>
            {tensor.numel
              ? "Select a cell. Arrow keys navigate the tensor; large dimensions are shown as a window."
              : "Empty tensor: no elements."}
          </title>
          {!tensor.numel ? (
            <text
              x={canvasWidth / 2}
              y={viewHeight / 2}
              textAnchor="middle"
              className="svg-axis"
            >
              Empty tensor · no elements
            </text>
          ) : (
            <>
              {Array.from(
                { length: Math.min(3, stacks) - 1 },
                (_, i) => Math.min(3, stacks) - 1 - i,
              ).map((layer) => (
                <rect
                  className="tensor-sheet"
                  key={layer}
                  x={startX + layer * 4}
                  y={startY - layer * 5}
                  width={width - 2}
                  height={height - 2}
                  rx="4"
                />
              ))}
              {Array.from({ length: visibleCols }, (_, column) => (
                <text
                  className="cell-coordinate"
                  key={column}
                  x={startX + (column + 0.5) * cellW - 1}
                  y={startY - 10}
                  textAnchor="middle"
                  style={{
                    fontSize: Math.min(
                      8,
                      28 / (String(column + colStart).length * 0.65),
                    ),
                  }}
                >
                  {column + colStart}
                </text>
              ))}
              {Array.from({ length: visibleRows }, (_, row) => (
                <text
                  className="cell-coordinate"
                  key={row}
                  x={startX - 9}
                  y={startY + (row + 0.5) * cellH + 2}
                  textAnchor="end"
                  style={{
                    fontSize: Math.min(
                      8,
                      34 / (String(row + rowStart).length * 0.65),
                    ),
                  }}
                >
                  {row + rowStart}
                </text>
              ))}
              {cells.map(({ row, column, coordinates, flat }) => {
                const value = data.valueAt(flat),
                  active = linked.has(flat),
                  pinned = flat === index,
                  burning = pinned && light === "active";
                const text = shapeOnly
                  ? "·"
                  : tensor.dtype === "bool" && value !== undefined
                    ? exactValue(value, "bool")
                    : formatCellValue(value, 8);
                const tint = burning ? null : (paint?.(tensor, flat) ?? null);
                const intensity =
                  showValues && typeof value === "number"
                    ? Math.min(0.28, (Math.abs(value) / magnitude) * 0.28)
                    : 0;
                const fontSize = Math.min(
                  8.5,
                  (cellW - 14) / (text.length * 0.58),
                );
                return (
                  <g
                    key={flat}
                    role="button"
                    tabIndex={pinned ? 0 : -1}
                    data-cell-index={flat}
                    aria-label={`${label} element ${coordinates.join(",") || "scalar"} ${shapeOnly ? "shape only" : `value ${formatValue(value)}`}`}
                    aria-pressed={pinned}
                    className={`tensor-cell ${burning ? "cell-selected" : pinned ? "cell-source" : ""} ${active ? "cell-linked" : ""} ${tint ? "cell-inked" : ""} ${tint && linked.size && !active ? "cell-quiet" : ""}`}
                    onClick={() => select(flat)}
                    onMouseEnter={() => setHovered(flat)}
                    onFocus={() => setHovered(null)}
                    onKeyDown={(event) => {
                      if (
                        [
                          "ArrowLeft",
                          "ArrowRight",
                          "ArrowUp",
                          "ArrowDown",
                          "Home",
                          "End",
                        ].includes(event.key)
                      ) {
                        event.preventDefault();
                        const next = moveInPlane(
                          flat,
                          tensor.shape,
                          plane,
                          event.key,
                        );
                        if (next !== index) {
                          keyboardFocus.current = true;
                          select(next);
                        }
                      } else if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        select(flat);
                      }
                    }}
                  >
                    <title>{`[${coordinates.join(", ")}]${shapeOnly ? " · shape only" : ` = ${String(value ?? "Loading value")}`}`}</title>
                    <rect
                      x={startX + column * cellW}
                      y={startY + row * cellH}
                      width={cellW - 2}
                      height={cellH - 2}
                      rx="3"
                      style={{
                        ...(tint
                          ? ({ "--cell-ink": tint } as React.CSSProperties)
                          : {}),
                        fill: burning
                          ? "url(#tv-fire)"
                          : tint
                            ? "var(--cell-ink)"
                            : active
                              ? "var(--tensor-linked)"
                              : `color-mix(in srgb, var(--tensor-color) ${5 + intensity * 100}%, var(--tensor-base))`,
                      }}
                    />
                    {(showValues || pinned) && (
                      <svg
                        x={startX + column * cellW + 5}
                        y={startY + row * cellH}
                        width={cellW - 12}
                        height={cellH - 2}
                        style={{ overflow: "hidden" }}
                        aria-hidden="true"
                      >
                        <text
                          x={(cellW - 12) / 2}
                          y={(cellH - 2) / 2}
                          dominantBaseline="central"
                          textAnchor="middle"
                          style={{ fontSize }}
                          textLength={Math.min(
                            cellW - 12,
                            text.length * fontSize * 0.6,
                          )}
                          lengthAdjust="spacingAndGlyphs"
                        >
                          {text}
                        </text>
                      </svg>
                    )}
                  </g>
                );
              })}
              <text
                x={startX + width / 2}
                y={startY + height + 20}
                textAnchor="middle"
                className="svg-axis"
                style={
                  plane.column !== null && ink?.[plane.column]
                    ? { fill: ink[plane.column]!.colors[0] }
                    : undefined
                }
              >
                {plane.column === null
                  ? "scalar"
                  : `${axisName(plane.column)} · ${columns.toLocaleString()}`}
              </text>
              {plane.row !== null && (
                <text
                  transform={`translate(12,${startY + height / 2}) rotate(-90)`}
                  textAnchor="middle"
                  className="svg-axis"
                  style={
                    ink?.[plane.row]
                      ? { fill: ink[plane.row]!.colors[0] }
                      : undefined
                  }
                >
                  {axisName(plane.row)} · {rows.toLocaleString()}
                </text>
              )}
            </>
          )}
        </svg>
      </div>
      {!!tensor.numel && (
        <div
          className="tensor-window-summary"
          aria-label={`${label} visible window`}
        >
          <span>
            {shapeOnly
              ? "Shape preview"
              : data.loading
                ? "Loading values…"
                : `${(visibleRows * visibleCols).toLocaleString()} of ${tensor.numel.toLocaleString()} elements`}
          </span>
          {(rows > pageSize || columns > pageSize) && (
            <small>
              {plane.row !== null
                ? `Rows ${rowStart.toLocaleString()}–${(rowStart + visibleRows - 1).toLocaleString()} · `
                : ""}
              Cols {colStart.toLocaleString()}–
              {(colStart + visibleCols - 1).toLocaleString()}
            </small>
          )}
        </div>
      )}
      {!!tensor.numel && (rows > pageSize || columns > pageSize) && (
        <div
          className="tensor-page-controls"
          role="group"
          aria-label={`${label} window navigation`}
        >
          {(["row", "column"] as const).map((side) => {
            const axis = plane[side];
            return axis !== null && tensor.shape[axis] > pageSize ? (
              <IndexControl
                key={side}
                label={`Go to ${side}`}
                name={`${label} ${side} coordinate`}
                value={coords[axis]}
                size={tensor.shape[axis]}
                step={pageSize}
                onChange={(value) => sliceAt(axis, value)}
              />
            ) : null;
          })}
        </div>
      )}
      {data.error && (
        <div className="tensor-load-error" role="alert">
          {data.error}
          <button className="text-button" onClick={data.retry}>
            Retry
          </button>
        </div>
      )}
      {!!tensor.numel && (
        <div className="tensor-readout" aria-label={`${label} element details`}>
          <span>{hovered === null ? "Selected" : "Preview"}</span>
          <code>[{readCoords.join(", ")}]</code>
          <strong
            title={`As stored in ${tensor.dtype}; grid labels are rounded.${typeof data.valueAt(readIndex) === "number" ? ` Recorded as ${data.valueAt(readIndex)}.` : ""}`}
          >
            {shapeOnly
              ? "Shape only"
              : data.valueAt(readIndex) !== undefined
                ? exactValue(data.valueAt(readIndex), tensor.dtype)
                : data.loading
                  ? "Loading…"
                  : "Not available"}
          </strong>
        </div>
      )}
      {!!tensor.numel && !shapeOnly && (
        <ValueSpread
          tensor={tensor}
          value={data.valueAt(readIndex) as number | string | undefined}
        />
      )}
      {!!tensor.numel && rank > 0 && (
        <CoordinateJump
          label={label}
          summaryLabel="Go to cell"
          shape={tensor.shape}
          coords={coords}
          onSelect={jumpTo}
        />
      )}
      <div
        id={`${id}-controls`}
        className="tensor-explore-panel"
        hidden={!settings}
      >
        <div className="explore-heading">
          <span>Viewing plane</span>
          <small>Changes the view, not the tensor</small>
        </div>
        <div className="plane-controls">
          {rank > 1 &&
            (["row", "column"] as const).map((side) => (
              <label key={side}>
                {side === "row" ? "Rows" : "Columns"}
                <select
                  aria-label={`${label} ${side} axis`}
                  value={plane[side] ?? ""}
                  onChange={(event) => {
                    setPlane(
                      changePlane(plane, side, Number(event.target.value)),
                    );
                    setHovered(null);
                  }}
                >
                  {tensor.shape.map((_, axis) => (
                    <option key={axis} value={axis}>
                      {axis} · {axisName(axis)}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          {(rows > 8 || columns > 8) && (
            <label>
              Window
              <select
                aria-label={`${label} window size`}
                value={pageSize}
                onChange={(event) => setPageSize(Number(event.target.value))}
              >
                <option value={8}>8 × 8</option>
                <option value={16}>16 × 16</option>
              </select>
            </label>
          )}
        </div>
      </div>
      <details className="memory-details" open={expandDetails}>
        <summary aria-label={`${label} tensor details`}>
          Details <span>{tensor.dtype}</span>
        </summary>
        <dl>
          <dt>Elements</dt>
          <dd>{tensor.numel.toLocaleString()}</dd>
          <dt>Data</dt>
          <dd>
            {shapeOnly
              ? "Shape metadata; no values"
              : tensor.value_source === "paged"
                ? "Values loaded by window"
                : "Recorded values"}
          </dd>
          <dt>Flat index</dt>
          <dd>{index.toLocaleString()}</dd>
          <dt>Storage</dt>
          <dd>
            {tensor.storage_id} · storage position{" "}
            {storagePosition(coords, tensor.strides, tensor.storage_offset)}
          </dd>
          <dt>Strides</dt>
          <dd>[{tensor.strides.join(", ")}]</dd>
          <dt>Contiguous</dt>
          <dd>{tensor.contiguous ? "Yes" : "No"}</dd>
        </dl>
      </details>
    </article>
  );
}

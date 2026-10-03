import {
  useContext,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Box,
  GitCompare,
  Image,
  SlidersHorizontal,
  Thermometer,
} from "lucide-react";
import { pixelPlan } from "../inputs/samples";
import { PixelView } from "./PixelView";
import { describeAxis, restatesAxis, shortAxis } from "./axisLineage";
import { useAxisOrigins } from "./LineageContext";
import { inkStyle, useAxisInk, useCellPaint } from "./InkShape";
import "./gridInk.css";
import type { Tensor } from "../api/client";
import {
  exactValue,
  formatCellValue,
  formatValue,
  product,
  pythonList,
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
import { elementBytes, formatBytes, tensorBytes } from "./memory";
import {
  isBroken,
  landmarks,
  nextBroken,
  planeValues,
  quantiles,
} from "./find";
import { TensorUseContext, type UseStep } from "./TensorUseContext";
import {
  heatFill,
  heatGradient,
  heatLevel,
  heatRangeOf,
  useDiff,
  useHeat,
  useWideWindow,
} from "./heat";
import {
  RunBeforeContext,
  cellDeltas,
  closeness,
  loadComparison,
  matchingState,
} from "./diff";
import { ValueFind } from "./ValueFind";
import { contractFor } from "../editor/valueContracts";
import { ContractContext } from "../editor/ContractContext";
import { AxisProfileChart } from "./AxisProfile";
import { CellHistory, namedStates } from "./CellHistory";
import { Filmstrip } from "./Filmstrip";
import { rangeCells, rangeList, rangeStats, rangeTsv } from "./range";
import {
  asNumber,
  binCells,
  planeMargins,
  type Margins,
  type Reduce,
} from "./margins";
import { useTensorQuery } from "./useTensorQuery";
import type { ChangeSummary, RegionResult, SearchResult } from "../api/client";

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
  // The window over a large plane, shared by every grid and remembered.
  const [wideWindow, setWideWindow] = useWideWindow();
  const pageSize = wideWindow ? 16 : 8;
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
  // Margins add a column of row totals and a row of column totals.
  const [marginKind, setMarginKind] = useState<Reduce | null>(null);
  const hiddenKey = slices.map((axis) => coords[axis]).join();
  // Large tensors keep their values in a snapshot; the backend answers
  // whole-tensor questions about them.
  const paged = tensor.value_source === "paged" && !!runId;
  // The plane through the cursor: its own axes at 0, the hidden ones fixed.
  const planeQuery = {
    row: plane.row ?? -1,
    column: plane.column ?? -1,
    fixed: coords
      .map((coordinate, axis) =>
        axis === plane.row || axis === plane.column ? 0 : coordinate,
      )
      .join(","),
  };
  const localMargins = useMemo(
    () =>
      marginKind && tensor.value_source !== "shape"
        ? planeMargins(tensor, plane, coords, marginKind)
        : null,
    // The plane through the cursor depends only on its hidden coordinates.
    [tensor, plane, hiddenKey, marginKind],
  );
  const remoteMargins = useTensorQuery<{
    rows: (number | string)[];
    columns: (number | string)[];
    all: number | string;
  }>(
    runId,
    tensor.id,
    "margins",
    paged && marginKind ? { ...planeQuery, reduce: marginKind } : null,
  );
  const margins: Margins | null =
    localMargins ??
    (remoteMargins.data && {
      rows: remoteMargins.data.rows.map(asNumber),
      columns: remoteMargins.data.columns.map(asNumber),
      all: asNumber(remoteMargins.data.all),
    });
  const marginW = margins ? cellW + 6 : 0,
    marginH = margins ? cellH + 4 : 0;
  const canvasWidth = frameCols * cellW + 94 + marginW,
    viewHeight = Math.max(110, frameRows * cellH + 70) + marginH;
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
  const [heat, setHeat] = useHeat();
  // Shading by value spans the whole tensor's range, so slices compare.
  const heatRange = heat ? heatRangeOf(tensor) : null;
  // Diff: each cell's change since the run before, which loads on demand.
  const [diffOn, setDiff] = useDiff();
  const runBefore = useContext(RunBeforeContext);
  useEffect(() => {
    if (diffOn && runBefore && !runBefore.before) runBefore.request();
  }, [diffOn, runBefore]);
  // Or another tensor of this run, chosen under Compare with, shown in the
  // grid the same way: each cell's difference from it.
  const peers = useContext(TensorUseContext);
  const [against, setAgainst] = useState<string | null>(null);
  const peerState =
    against && peers?.trace
      ? ((peers.trace.tensors[against] as Tensor | undefined) ?? null)
      : null;
  const beforeState = useMemo(
    () =>
      peerState ??
      (diffOn && runBefore?.before
        ? matchingState(runBefore.run.trace, runBefore.before.trace, tensor.id)
        : null),
    [peerState, diffOn, runBefore, tensor.id],
  );
  const vsLabel = peerState ? peerState.name : "the run before";
  const localDiff = useMemo(
    () => (beforeState ? cellDeltas(tensor, beforeState) : null),
    [beforeState, tensor],
  );
  // Values kept in snapshots are compared by the backend, and the cells on
  // screen read their earlier values window by window.
  const thisRunId = peerState ? (peers?.runId ?? runId) : runBefore?.run.id;
  const earlierId = peerState ? thisRunId : runBefore?.before?.id;
  const compareKey =
    beforeState && !localDiff && earlierId && thisRunId
      ? `${thisRunId}|${earlierId}|${tensor.id}|${beforeState.id}`
      : null;
  const [compared, setCompared] = useState<{
    key: string;
    summary: ChangeSummary | null;
  } | null>(null);
  useEffect(() => {
    if (!compareKey) return;
    const [runKey, earlierKey, nowId, thenId] = compareKey.split("|");
    let current = true;
    loadComparison(runKey, earlierKey, [[nowId, thenId]])
      .then(([summary]) => current && setCompared({ key: compareKey, summary }))
      .catch(() => current && setCompared({ key: compareKey, summary: null }));
    return () => {
      current = false;
    };
  }, [compareKey]);
  const remoteSummary =
    compared && compared.key === compareKey ? compared.summary : null;
  const diff = localDiff
    ? localDiff
    : remoteSummary && {
        low: asNumber(remoteSummary.low),
        high: asNumber(remoteSummary.high),
        changed: remoteSummary.changed,
        compared: remoteSummary.compared,
      };
  const remoteBefore = !!compareKey;
  const beforeValues = useTensorValues(
    beforeState ?? tensor,
    remoteBefore ? earlierId : undefined,
    remoteBefore ? cells.map((cell) => cell.flat) : [],
  );
  const beforeAt = (flat: number) =>
    !beforeState
      ? undefined
      : localDiff
        ? beforeState.values[flat]
        : beforeValues.valueAt(flat);
  /** A cell's change since the run before; null when it has no size. */
  const changeAt = (flat: number, value: number | string | undefined) => {
    if (localDiff) return localDiff.deltas[flat];
    const then = beforeAt(flat);
    if (value === undefined || then === undefined) return null;
    if (isBroken(value) || isBroken(then)) return null;
    return Number(value) - Number(then);
  };
  const reach = diff ? Math.max(-diff.low, diff.high) : 0;
  const diffNote = peerState
    ? !diff
      ? compareKey && compared?.key !== compareKey
        ? `Comparing with ${peerState.name}…`
        : `${peerState.name} could not be compared with this tensor.`
      : null
    : !diffOn
      ? null
      : !runBefore
        ? null
        : !runBefore.before
          ? "Loading the run before…"
          : !beforeState
            ? "No matching state in the run before: the step, shape or dtype differs."
            : !diff
              ? compareKey && compared?.key !== compareKey
                ? "Comparing with the run before…"
                : "The run before could not be compared with this tensor."
              : null;
  const [matches, setMatches] = useState<Set<number> | null>(null);
  // Every slice of the innermost hidden axis as a picture, on request.
  const [filmOpen, setFilmOpen] = useState(false);
  const filmAxis = slices.filter((axis) => tensor.shape[axis] > 1).at(-1);
  // Details computes its heavier parts, such as the axis profile, only open.
  const [detailsOpen, setDetailsOpen] = useState(expandDetails);
  // Shift extends a rectangle from an anchor cell, like a spreadsheet.
  const [anchor, setAnchor] = useState<number | null>(null);
  // A histogram bar, chosen to ring the cells whose values fall in it.
  const [bin, setBin] = useState<{
    index: number;
    from: number;
    to: number;
    last: boolean;
  } | null>(null);
  const localBin = useMemo(
    () =>
      bin && tensor.histogram
        ? binCells(
            tensor,
            tensor.histogram.low,
            tensor.histogram.high,
            tensor.histogram.counts.length,
            bin.index,
          )
        : null,
    [bin, tensor],
  );
  const remoteBin = useTensorQuery<SearchResult>(
    runId,
    tensor.id,
    "search",
    bin && paged
      ? { test: "between", value: bin.from, high: bin.to, closed: bin.last }
      : null,
  );
  const binFound = localBin?.cells ?? remoteBin.data?.indices ?? null;
  const binCount = localBin?.cells.length ?? remoteBin.data?.count ?? null;
  const ringed = bin ? new Set(binFound ?? []) : matches;
  const root = useRef<HTMLElement>(null);
  const shapeOnly = tensor.value_source === "shape";
  const bytes = tensorBytes(tensor);
  const localMarks = useMemo(
    () => (shapeOnly ? null : landmarks(tensor)),
    [tensor, shapeOnly],
  );
  const remoteMarks = useTensorQuery<{
    min: number | null;
    max: number | null;
    broken: number[];
    quantiles: (number | string)[] | null;
  }>(runId, tensor.id, "landmarks", paged && !localMarks ? {} : null);
  const marks = localMarks ?? remoteMarks.data;
  const flow = useContext(TensorUseContext);
  const contracts = useContext(ContractContext);
  const contract = contracts?.byTensor.get(tensor.id);
  const uses = flow?.uses(tensor.id) ?? null;
  const sliceName = `${tensor.name}[${coords.map((coordinate, axis) => (axis === plane.row || axis === plane.column ? ":" : coordinate)).join(", ")}]`;
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
  const range =
    anchor !== null && tensor.numel
      ? rangeCells(tensor.shape, plane, anchor, index)
      : null;
  const inRange = range ? new Set(range.cells) : null;
  const inline = tensor.values.length === tensor.numel;
  const rangeValues =
    range && inline && !shapeOnly
      ? range.cells.map((flat) => tensor.values[flat])
      : null;
  const region = useTensorQuery<RegionResult>(
    runId,
    tensor.id,
    "region",
    range && paged
      ? {
          ...planeQuery,
          rows: range.rowSpan.join(","),
          columns: range.columnSpan.join(","),
        }
      : null,
    150,
  );

  /**
   * Select a whole row or column of the plane, or all of it, as a rectangle
   * from its far end back to its first cell, so the window shows its start.
   */
  function selectLine(kind: "row" | "column" | "all", at: number) {
    if (!tensor.numel) return;
    const lastRow = plane.row === null ? 0 : tensor.shape[plane.row] - 1;
    const lastColumn =
      plane.column === null ? 0 : tensor.shape[plane.column] - 1;
    const cell = (row: number, column: number) =>
      ravel(
        planeCoordinates(tensor.shape, plane, coords, row, column),
        tensor.shape,
      );
    const [far, near] =
      kind === "column"
        ? [cell(lastRow, at), cell(0, at)]
        : kind === "row"
          ? [cell(at, lastColumn), cell(at, 0)]
          : [cell(lastRow, lastColumn), cell(0, 0)];
    setAnchor(far === near ? null : far);
    select(near);
  }
  /** Open a panel under the grid and put the caret in its field. */
  function openPanel(selector: string) {
    const panel = root.current?.querySelector<HTMLDetailsElement>(selector);
    if (!panel) return;
    panel.open = true;
    requestAnimationFrame(() => {
      const input = panel.querySelector<HTMLInputElement>("input");
      input?.focus();
      input?.select();
    });
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
      ref={root}
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
          {runBefore && !shapeOnly && !!tensor.numel && (
            <button
              className="tensor-view-toggle"
              aria-label="Show the change since the run before"
              aria-pressed={diffOn}
              onClick={() => setDiff(!diffOn)}
              title="Show each cell's change since the run before: blue fell, rose rose, plain unchanged. Applies to every grid."
            >
              <GitCompare size={14} />
              <span>Diff</span>
            </button>
          )}
          {!shapeOnly && !!tensor.numel && (
            <button
              className="tensor-view-toggle"
              aria-label="Shade cells by value"
              aria-pressed={heat}
              onClick={() => setHeat(!heat)}
              title={
                heat
                  ? "Cells are shaded by value: blue below zero, rose above, across the whole tensor's range. Applies to every grid."
                  : "Shade cells by value, like a heatmap. Applies to every grid."
              }
            >
              <Thermometer size={14} />
              <span>Heat</span>
            </button>
          )}
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
          {(rank > 1 ||
            rows > 8 ||
            columns > 8 ||
            (tensor.numel > 1 && !shapeOnly)) && (
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
                {short && !restatesAxis(origin!, axisName(axis)) && (
                  <i className="axis-origin">← {short}</i>
                )}
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
          {filmAxis !== undefined && !shapeOnly && (
            <button
              type="button"
              className="text-button filmstrip-toggle"
              aria-expanded={filmOpen}
              title={`Picture every ${axisName(filmAxis)} slice at once`}
              onClick={() => setFilmOpen(!filmOpen)}
            >
              {filmOpen ? "Hide" : "All"} {tensor.shape[filmAxis]}
            </button>
          )}
        </div>
      )}
      {filmOpen && filmAxis !== undefined && !shapeOnly && !!tensor.numel && (
        <Filmstrip
          tensor={tensor}
          runId={runId}
          plane={plane}
          coords={coords}
          axis={filmAxis}
          axisName={axisName(filmAxis)}
          onPick={(value) => sliceAt(filmAxis, value)}
        />
      )}
      <div
        className="tensor-drawing"
        style={
          { "--tensor-grid-height": `${viewHeight}px` } as React.CSSProperties
        }
      >
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
              ? `Select a cell. Arrow keys navigate the tensor; large dimensions are shown as a window.${slices.some((axis) => tensor.shape[axis] > 1) ? " Page Up and Page Down step through slices." : ""}`
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
                  className={`cell-coordinate cell-header${plane.column !== null && column + colStart === readCoords[plane.column] ? " cell-coordinate-active" : ""}`}
                  key={column}
                  onClick={() => selectLine("column", column + colStart)}
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
                  className={`cell-coordinate cell-header${plane.row !== null && row + rowStart === readCoords[plane.row] ? " cell-coordinate-active" : ""}`}
                  key={row}
                  onClick={() => selectLine("row", row + rowStart)}
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
                const change = diff ? changeAt(flat, value) : null;
                // NaN or infinity appearing or going away is a change with
                // no size: it shades as fully changed.
                const then = diff ? beforeAt(flat) : undefined;
                const flipped =
                  !!diff &&
                  change === null &&
                  value !== undefined &&
                  then !== undefined &&
                  isBroken(value) !== isBroken(then);
                const text = shapeOnly
                  ? "·"
                  : change !== null
                    ? change === 0
                      ? "0"
                      : `${change > 0 ? "+" : ""}${formatCellValue(change, 7)}`
                    : tensor.dtype === "bool" && value !== undefined
                      ? exactValue(value, "bool")
                      : formatCellValue(value, 8);
                // A diff shades by change; otherwise heat shades by value.
                const level = diff
                  ? flipped
                    ? 1
                    : change !== null && change !== 0 && reach
                      ? change / reach
                      : null
                  : heatRange && typeof value === "number"
                    ? heatLevel(value, heatRange.low, heatRange.high)
                    : null;
                const tint =
                  burning || level !== null
                    ? null
                    : (paint?.(tensor, flat) ?? null);
                const intensity =
                  showValues &&
                  typeof value === "number" &&
                  Number.isFinite(value)
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
                    className={`tensor-cell ${burning ? "cell-selected" : pinned ? "cell-source" : ""} ${active ? "cell-linked" : ""} ${tint ? "cell-inked" : ""} ${tint && linked.size && !active ? "cell-quiet" : ""}${isBroken(value) ? " cell-broken" : ""}${level !== null ? " cell-heat" : ""}${diff && change === 0 ? " cell-unchanged" : ""}${ringed?.has(flat) ? " cell-match" : ""}${inRange?.has(flat) ? " cell-range" : ""}`}
                    onClick={(event) => {
                      setAnchor(event.shiftKey ? (anchor ?? index) : null);
                      select(flat);
                    }}
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
                          "PageUp",
                          "PageDown",
                        ].includes(event.key)
                      ) {
                        event.preventDefault();
                        // Shift with arrows or Home/End grows the selection;
                        // a new slice ends it.
                        setAnchor(
                          event.shiftKey && !event.key.startsWith("Page")
                            ? (anchor ?? flat)
                            : null,
                        );
                        const corner =
                          (event.ctrlKey || event.metaKey) &&
                          (event.key === "Home" || event.key === "End");
                        const next = moveInPlane(
                          flat,
                          tensor.shape,
                          plane,
                          corner
                            ? event.key === "Home"
                              ? "PlaneStart"
                              : "PlaneEnd"
                            : event.key,
                        );
                        if (next !== index) {
                          keyboardFocus.current = true;
                          select(next);
                        }
                      } else if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        select(flat);
                      } else if (
                        (event.ctrlKey || event.metaKey) &&
                        event.key.toLowerCase() === "a"
                      ) {
                        // The whole plane on screen, as a selection.
                        event.preventDefault();
                        keyboardFocus.current = true;
                        selectLine("all", 0);
                      } else if (event.key === "Escape" && range) {
                        event.preventDefault();
                        event.stopPropagation();
                        setAnchor(null);
                      } else if (
                        (event.ctrlKey || event.metaKey) &&
                        event.key.toLowerCase() === "c" &&
                        !shapeOnly
                      ) {
                        // Copy the selection as spreadsheet rows, or the cell.
                        const block = rangeValues ?? region.data?.values;
                        const text =
                          range && block
                            ? rangeTsv(block, { columns: range.columns })
                            : value !== undefined
                              ? exactValue(value, tensor.dtype)
                              : null;
                        if (text !== null) {
                          event.preventDefault();
                          void navigator.clipboard
                            ?.writeText(text)
                            .catch(() => {});
                        }
                      } else if (
                        ((event.ctrlKey || event.metaKey) &&
                          event.key.toLowerCase() === "f") ||
                        (event.key === "/" && !event.ctrlKey && !event.metaKey)
                      ) {
                        // Find in this tensor, not the page.
                        event.preventDefault();
                        openPanel(".value-find");
                      } else if (
                        event.key === "g" &&
                        !event.ctrlKey &&
                        !event.metaKey &&
                        !event.altKey
                      ) {
                        event.preventDefault();
                        openPanel(".coordinate-jump:not(.value-find)");
                      } else if (
                        (event.key === "w" || event.key === "W") &&
                        !event.ctrlKey &&
                        !event.metaKey &&
                        !event.altKey &&
                        (rows > 8 || columns > 8)
                      ) {
                        // An 8 × 8 or 16 × 16 window, in every grid.
                        event.preventDefault();
                        keyboardFocus.current = true;
                        setWideWindow(!wideWindow);
                      } else if (
                        (event.key === "t" || event.key === "T") &&
                        !event.ctrlKey &&
                        !event.metaKey &&
                        plane.row !== null &&
                        plane.column !== null
                      ) {
                        // Swap the axes on screen; the tensor is unchanged.
                        event.preventDefault();
                        keyboardFocus.current = true;
                        setPlane({ row: plane.column, column: plane.row });
                      } else if (
                        (event.key === "[" || event.key === "]") &&
                        flow?.trace
                      ) {
                        // The previous or next state of the same name.
                        const states = namedStates(flow.trace, tensor);
                        const at = states.findIndex(
                          (state) => state.id === tensor.id,
                        );
                        const next = states[at + (event.key === "]" ? 1 : -1)];
                        const step = next && flow.uses(next.id).made;
                        if (step) {
                          event.preventDefault();
                          event.stopPropagation();
                          flow.go(step.id);
                        }
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
                          : isBroken(value)
                            ? "color-mix(in srgb, var(--danger, #efa99b) 22%, var(--tensor-base))"
                            : level !== null
                              ? heatFill(level)
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
              {margins && (
                <MarginCells
                  kind={marginKind!}
                  margins={margins}
                  rowStart={rowStart}
                  colStart={colStart}
                  visibleRows={visibleRows}
                  visibleCols={visibleCols}
                  left={startX}
                  top={startY}
                  width={width}
                  height={height}
                  cellW={cellW}
                  cellH={cellH}
                />
              )}
              <text
                x={startX + width / 2}
                y={startY + height + 20 + marginH}
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
      {diff && !!tensor.numel && (
        <div
          className="tensor-heat-legend"
          role="img"
          aria-label={
            diff.changed
              ? `Difference from ${vsLabel}, from ${formatValue(diff.low)} to ${formatValue(diff.high)}: ${diff.changed} of ${diff.compared} values differ`
              : `No value differs from ${vsLabel}`
          }
        >
          {diff.changed ? (
            <>
              <span>Δ {formatValue(-reach)}</span>
              <i style={{ background: heatGradient(-reach, reach) }}>
                <b style={{ left: "50%" }} title="No change" />
              </i>
              <span>+{formatValue(reach)}</span>
              <small>
                {diff.changed.toLocaleString()} of{" "}
                {diff.compared.toLocaleString()}{" "}
                {peerState ? `differ from ${peerState.name}` : "changed"}
              </small>
            </>
          ) : (
            <small>No value differs from {vsLabel}</small>
          )}
        </div>
      )}
      {diffNote && !!tensor.numel && (
        <p className="tensor-diff-note">{diffNote}</p>
      )}
      {heatRange && !diff && !!tensor.numel && (
        <div
          className="tensor-heat-legend"
          role="img"
          aria-label={`Shading from ${formatValue(heatRange.low)} to ${formatValue(heatRange.high)}${heatRange.low < 0 && heatRange.high > 0 ? ", centered on zero" : ""}`}
        >
          <span>{formatValue(heatRange.low)}</span>
          <i
            style={{ background: heatGradient(heatRange.low, heatRange.high) }}
          >
            {heatRange.low < 0 && heatRange.high > 0 && (
              <b
                style={{
                  left: `${(-heatRange.low / (heatRange.high - heatRange.low)) * 100}%`,
                }}
                title="Zero"
              />
            )}
          </i>
          <span>{formatValue(heatRange.high)}</span>
        </div>
      )}
      {range && (
        <RangeBar
          label={label}
          tensor={tensor}
          rows={range.rows}
          columns={range.columns}
          values={rangeValues ?? region.data?.values ?? null}
          remote={paged ? region : null}
          onClear={() => setAnchor(null)}
        />
      )}
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
          {(rows > 8 || columns > 8) && !shapeOnly && (
            <button
              type="button"
              className="tensor-window-size"
              aria-pressed={wideWindow}
              title={
                wideWindow
                  ? "Showing a 16 × 16 window in every grid; click for 8 × 8, with larger cells"
                  : "Show a 16 × 16 window of the plane in every grid"
              }
              onClick={() => setWideWindow(!wideWindow)}
            >
              16 × 16
            </button>
          )}
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
          <ReadoutNote
            tensor={tensor}
            value={data.valueAt(readIndex)}
            before={diff ? beforeAt(readIndex) : undefined}
            beforeLabel={peerState?.name}
          />
        </div>
      )}
      {!!tensor.numel && !shapeOnly && flow?.trace && (
        <CellHistory
          tensor={tensor}
          trace={flow.trace}
          runId={flow.runId ?? runId}
          index={index}
          stepOf={(id) => flow.uses(id).made}
          onOpen={flow.go}
        />
      )}
      {!!tensor.numel && !shapeOnly && (
        <ValueSpread
          tensor={tensor}
          value={data.valueAt(readIndex) as number | string | undefined}
          bin={bin?.index ?? null}
          before={diff && beforeState ? beforeState.histogram : null}
          onBin={
            marks
              ? (chosen) => {
                  const { low, high, counts } = tensor.histogram!;
                  const width = (high - low) / counts.length;
                  setBin(
                    bin?.index === chosen
                      ? null
                      : {
                          index: chosen,
                          from: low + chosen * width,
                          to:
                            chosen === counts.length - 1
                              ? high
                              : low + (chosen + 1) * width,
                          last: chosen === counts.length - 1,
                        },
                  );
                }
              : undefined
          }
          onFind={
            marks
              ? {
                  min:
                    marks.min === null ? undefined : () => jumpTo(marks.min!),
                  max:
                    marks.max === null ? undefined : () => jumpTo(marks.max!),
                  broken: marks.broken.length
                    ? () => jumpTo(nextBroken(marks.broken, index)!)
                    : undefined,
                }
              : undefined
          }
        />
      )}
      {bin && (
        <p className="tensor-bin-note" role="status">
          {binCount === null ? (
            "Finding the values in this bar…"
          ) : (
            <>
              Ringed: {binCount.toLocaleString()}{" "}
              {binCount === 1 ? "value" : "values"} from {formatValue(bin.from)}{" "}
              to {formatValue(bin.to)}
            </>
          )}
          {!!binFound?.length && (
            <button
              type="button"
              className="text-button"
              onClick={() => jumpTo(nextBroken(binFound, index)!)}
            >
              Next
            </button>
          )}
          <button
            type="button"
            className="text-button"
            onClick={() => setBin(null)}
          >
            Clear
          </button>
        </p>
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
      {!!tensor.numel && !shapeOnly && tensor.dtype !== "bool" && (
        <ValueFind
          label={label}
          tensor={tensor}
          runId={runId}
          index={index}
          onSelect={jumpTo}
          onMatches={setMatches}
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
          {tensor.numel > 1 && !shapeOnly && (
            <label>
              Margins
              <select
                aria-label={`${label} margins`}
                value={marginKind ?? ""}
                title="A column of row totals and a row of column totals, over the plane's full axes"
                onChange={(event) =>
                  setMarginKind((event.target.value || null) as Reduce | null)
                }
              >
                <option value="">None</option>
                <option value="sum">Sum</option>
                <option value="mean">Mean</option>
                <option value="max">Max</option>
                <option value="min">Min</option>
              </select>
            </label>
          )}
          {(rows > 8 || columns > 8) && (
            <label>
              Window
              <select
                aria-label={`${label} window size`}
                value={pageSize}
                onChange={(event) =>
                  setWideWindow(Number(event.target.value) === 16)
                }
              >
                <option value={8}>8 × 8</option>
                <option value={16}>16 × 16</option>
              </select>
            </label>
          )}
        </div>
      </div>
      <details
        className="memory-details"
        open={expandDetails}
        onToggle={(event) => setDetailsOpen(event.currentTarget.open)}
      >
        <summary aria-label={`${label} tensor details`}>
          Details{" "}
          <span>
            {tensor.dtype}
            {bytes !== null && ` · ${formatBytes(bytes)}`}
          </span>
        </summary>
        <dl>
          <dt>Elements</dt>
          <dd>{tensor.numel.toLocaleString()}</dd>
          {bytes !== null && (
            <>
              <dt>Memory</dt>
              <dd
                title={`${tensor.numel.toLocaleString()} elements × ${elementBytes(tensor.dtype)} bytes; views may share a larger storage`}
              >
                {formatBytes(bytes)}
                {bytes >= 1024 && ` · ${bytes.toLocaleString()} bytes`}
              </dd>
            </>
          )}
          {!shapeOnly && !!tensor.numel && (
            <ValueStats
              tensor={tensor}
              remoteQuantiles={remoteMarks.data?.quantiles ?? null}
            />
          )}
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
          {flow?.trace && !shapeOnly && !!tensor.numel && (
            <CompareWith
              tensor={tensor}
              flow={flow}
              shown={against}
              onShow={setAgainst}
            />
          )}
          {contract && (
            <>
              <dt>Contract</dt>
              <dd
                className={
                  contract.ok ? "tensor-compare-same" : "tensor-compare-differs"
                }
                title={contract.message}
              >
                {contract.ok ? "✓" : "✗"} line {contract.line}: {contract.text}
                {!contract.ok && <small> · {contract.message}</small>}
                {!contract.ok && contracts?.accept && (
                  <button
                    type="button"
                    className="text-button tensor-step-link"
                    title="Rewrite the broken clauses around what this run recorded"
                    onClick={() => contracts.accept!(contract.line)}
                  >
                    Accept recorded values
                  </button>
                )}
              </dd>
            </>
          )}
          {uses && flow && (
            <>
              <dt>Made by</dt>
              <dd>
                {uses.made ? (
                  <StepLink step={uses.made} go={flow.go} />
                ) : tensor.role === "parameter" ? (
                  "A parameter of the model"
                ) : (
                  "An input of the run"
                )}
              </dd>
              <dt>Read by</dt>
              <dd className="tensor-step-links">
                {uses.read.length
                  ? uses.read
                      .slice(0, 8)
                      .map((step) => (
                        <StepLink key={step.id} step={step} go={flow.go} />
                      ))
                  : "No later step"}
                {uses.read.length > 8 && (
                  <span>+{uses.read.length - 8} more</span>
                )}
              </dd>
            </>
          )}
        </dl>
        {detailsOpen && !shapeOnly && tensor.numel > 1 && rank > 1 && (
          <AxisProfileChart
            tensor={tensor}
            runId={runId}
            coords={coords}
            axisName={axisName}
            onPick={sliceAt}
          />
        )}
        {!shapeOnly && !!tensor.numel && (
          <CopyActions
            value={data.valueAt(readIndex)}
            tensor={tensor}
            runId={runId}
            coords={readCoords}
            slice={
              rank > 2 && rows * columns <= 10_000 && localMarks
                ? {
                    name: sliceName,
                    build: () => {
                      const values = planeValues(tensor, plane, coords);
                      return values && pythonList(values);
                    },
                  }
                : undefined
            }
          />
        )}
      </details>
    </article>
  );
}

const statistic = (value: number) =>
  Number.isInteger(value)
    ? value.toLocaleString()
    : String(Number(value.toPrecision(4)));

/** The whole tensor's exact range and summary statistics, as recorded. */
function ValueStats({
  tensor,
  remoteQuantiles,
}: {
  tensor: Tensor;
  /** From the backend, for values kept in a snapshot. */
  remoteQuantiles: (number | string)[] | null;
}) {
  const { minimum, maximum, histogram } = tensor;
  const local = useMemo(
    () =>
      tensor.values.length === tensor.numel && tensor.dtype !== "bool"
        ? quantiles(tensor.values)
        : null,
    [tensor],
  );
  const percentiles = local ?? remoteQuantiles?.map(asNumber) ?? null;
  const total = tensor.numel;
  const zeros = histogram?.zeros ?? 0;
  const broken = histogram?.non_finite ?? 0;
  const share = (count: number) => {
    const percent = (count / total) * 100;
    return percent > 0 && percent < 1 ? "<1%" : `${Math.round(percent)}%`;
  };
  return (
    <>
      {typeof minimum === "number" && typeof maximum === "number" && (
        <>
          <dt>Range</dt>
          <dd>
            {exactValue(minimum, tensor.dtype)} …{" "}
            {exactValue(maximum, tensor.dtype)}
          </dd>
        </>
      )}
      {typeof histogram?.mean === "number" && (
        <>
          <dt>Mean</dt>
          <dd>
            {statistic(histogram.mean)}
            {typeof histogram.std === "number" &&
              ` · σ ${statistic(histogram.std)}`}
          </dd>
        </>
      )}
      {percentiles && (
        <>
          <dt>Percentiles</dt>
          <dd title="1st, 50th (median) and 99th percentiles of the finite values, interpolated as numpy.percentile does">
            p1 {statistic(percentiles[0])} · median {statistic(percentiles[1])}{" "}
            · p99 {statistic(percentiles[2])}
          </dd>
        </>
      )}
      {histogram && (
        <>
          <dt>Zeros</dt>
          <dd>
            {zeros.toLocaleString()} · {share(zeros)}
          </dd>
        </>
      )}
      {broken > 0 && (
        <>
          <dt>NaN or ∞</dt>
          <dd className="memory-details-broken">
            {broken.toLocaleString()} · {share(broken)}
          </dd>
        </>
      )}
    </>
  );
}

const MARGIN_LABEL: Record<Reduce, string> = {
  sum: "Σ",
  mean: "mean",
  max: "max",
  min: "min",
};

/**
 * The margins of the grid: each visible row's reduction across all its
 * columns at the right, each visible column's down all its rows below, and
 * the whole plane's in the corner.
 */
function MarginCells({
  kind,
  margins,
  rowStart,
  colStart,
  visibleRows,
  visibleCols,
  left,
  top,
  width,
  height,
  cellW,
  cellH,
}: {
  kind: Reduce;
  margins: { rows: number[]; columns: number[]; all: number };
  rowStart: number;
  colStart: number;
  visibleRows: number;
  visibleCols: number;
  left: number;
  top: number;
  width: number;
  height: number;
  cellW: number;
  cellH: number;
}) {
  const x = left + width + 6,
    y = top + height + 4;
  const cell = (
    key: string,
    cx: number,
    cy: number,
    value: number,
    what: string,
  ) => {
    const text = formatCellValue(value, 7);
    return (
      <g
        key={key}
        className="cell-margin"
        aria-label={`${what}: ${formatValue(value)}`}
        role="img"
      >
        <title>{`${what} = ${value}`}</title>
        <rect x={cx} y={cy} width={cellW - 2} height={cellH - 2} rx="3" />
        <text
          x={cx + (cellW - 2) / 2}
          y={cy + (cellH - 2) / 2}
          dominantBaseline="central"
          textAnchor="middle"
          style={{
            fontSize: Math.min(8.5, (cellW - 10) / (text.length * 0.58)),
          }}
        >
          {text}
        </text>
      </g>
    );
  };
  const label = MARGIN_LABEL[kind];
  return (
    <g className="grid-margins">
      <text
        className="cell-coordinate margin-label"
        x={x + (cellW - 2) / 2}
        y={top - 10}
        textAnchor="middle"
      >
        {label}
      </text>
      <text
        className="cell-coordinate margin-label"
        x={left - 9}
        y={y + cellH / 2}
        textAnchor="end"
      >
        {label}
      </text>
      {Array.from({ length: visibleRows }, (_, row) =>
        cell(
          `r${row}`,
          x,
          top + row * cellH,
          margins.rows[row + rowStart],
          `${kind} of row ${row + rowStart}`,
        ),
      )}
      {Array.from({ length: visibleCols }, (_, column) =>
        cell(
          `c${column}`,
          left + column * cellW,
          y,
          margins.columns[column + colStart],
          `${kind} of column ${column + colStart}`,
        ),
      )}
      {cell("all", x, y, margins.all, `${kind} of the whole plane`)}
    </g>
  );
}

/**
 * A selected rectangle of cells: its size, the sum, mean and extremes of
 * its values, and copies for a spreadsheet or for Python.
 */
function RangeBar({
  label,
  tensor,
  rows,
  columns,
  values,
  remote,
  onClear,
}: {
  label: string;
  tensor: Tensor;
  rows: number;
  columns: number;
  values: (number | string | boolean | undefined)[] | null;
  /** Totals from the backend, for a tensor whose values stay in a snapshot. */
  remote?: {
    data: RegionResult | null;
    loading: boolean;
    error: string;
  } | null;
  onClear: () => void;
}) {
  const [copied, setCopied] = useState<"tsv" | "list" | null>(null);
  const stats = remote
    ? remote.data && {
        count: remote.data.count,
        sum: asNumber(remote.data.sum ?? undefined),
        mean: asNumber(remote.data.mean ?? undefined),
        std: asNumber(remote.data.std ?? undefined),
        min: asNumber(remote.data.min ?? undefined),
        max: asNumber(remote.data.max ?? undefined),
        broken: remote.data.broken,
      }
    : values
      ? rangeStats(values)
      : null;
  const count = rows * columns;
  async function copy(kind: "tsv" | "list", text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(kind);
      window.setTimeout(() => setCopied(null), 1400);
    } catch {
      setCopied(null);
    }
  }
  return (
    <div
      className="tensor-range"
      role="status"
      aria-label={`${label} selection`}
    >
      <span>
        <b>
          {rows} × {columns}
        </b>{" "}
        = {count.toLocaleString()} cells
      </span>
      {stats && stats.count > 0 && (
        <>
          <span>sum {formatValue(stats.sum)}</span>
          <span>mean {formatValue(stats.mean)}</span>
          {Number.isFinite(stats.std) && (
            <span>σ {formatValue(stats.std)}</span>
          )}
          <span>
            {formatValue(stats.min)} … {formatValue(stats.max)}
          </span>
        </>
      )}
      {stats && stats.broken > 0 && (
        <span className="tensor-range-broken">
          {stats.broken} NaN or ∞ left out
        </span>
      )}
      {remote?.loading && <span>Adding up…</span>}
      {remote?.error && <span>{remote.error}</span>}
      {!values && !remote?.loading && (
        <span>
          {remote
            ? "Too many cells to copy; select up to 10,000."
            : "Values load by window; copy and totals need recorded values."}
        </span>
      )}
      <span className="tensor-range-actions">
        {values && (
          <>
            <button
              type="button"
              className="text-button"
              title="Copy as tab-separated rows, which paste into a spreadsheet as cells"
              onClick={() => void copy("tsv", rangeTsv(values, { columns }))}
            >
              {copied === "tsv" ? "Copied" : "Copy TSV"}
            </button>
            <button
              type="button"
              className="text-button"
              title="Copy as a nested Python list, rows of columns"
              onClick={() =>
                void copy(
                  "list",
                  rangeList(values, { rows, columns }, tensor.dtype),
                )
              }
            >
              {copied === "list" ? "Copied" : "Copy list"}
            </button>
          </>
        )}
        <button
          type="button"
          className="text-button"
          title="Clear the selection (Escape)"
          onClick={onClear}
        >
          Clear
        </button>
      </span>
    </div>
  );
}

/**
 * Beside the readout value: how far it sits from the mean in standard
 * deviations, and with a diff on, what it was in the run before.
 */
function ReadoutNote({
  tensor,
  value,
  before,
  beforeLabel,
}: {
  tensor: Tensor;
  value: number | string | undefined;
  before: number | string | boolean | undefined;
  /** The tensor `before` comes from, when not the run before. */
  beforeLabel?: string;
}) {
  const mean = tensor.histogram?.mean;
  const std = tensor.histogram?.std;
  const x = typeof value === "number" ? value : NaN;
  const z =
    typeof mean === "number" && typeof std === "number" && std > 0
      ? (x - mean) / std
      : NaN;
  if (!Number.isFinite(z) && before === undefined) return null;
  return (
    <small className="tensor-readout-note">
      {Number.isFinite(z) && (
        <span
          className={Math.abs(z) >= 3 ? "tensor-readout-outlier" : undefined}
          title={`${Math.abs(z).toFixed(2)} standard deviations ${z < 0 ? "below" : "above"} the tensor's mean${Math.abs(z) >= 3 ? "; an outlier" : ""}`}
        >
          {z >= 0 ? "+" : "−"}
          {Math.abs(z).toFixed(1)}σ
        </span>
      )}
      {before !== undefined && (
        <span
          title={
            beforeLabel
              ? `The same cell of ${beforeLabel}`
              : "The same cell in the run before"
          }
        >
          {beforeLabel ? `${beforeLabel}:` : "was"}{" "}
          {exactValue(before, tensor.dtype)}
        </span>
      )}
    </small>
  );
}

/**
 * Compare this tensor with another recorded state of the same shape: how
 * many values differ, by how much, and whether torch.allclose would pass.
 */
function CompareWith({
  tensor,
  flow,
  shown,
  onShow,
}: {
  tensor: Tensor;
  flow: {
    uses: (id: string) => { made: UseStep | null };
    trace?: { tensors: Record<string, unknown> };
    runId?: string;
  };
  /** The tensor whose differences the grid shows, if any. */
  shown: string | null;
  onShow: (id: string | null) => void;
}) {
  const [other, setOther] = useState(shown ?? "");
  const tensors = (flow.trace?.tensors ?? {}) as Record<string, Tensor>;
  const shape = tensor.shape.join();
  const candidates = useMemo(
    () =>
      Object.values(tensors)
        .filter(
          (state) =>
            state.id !== tensor.id &&
            state.shape.join() === shape &&
            state.value_source !== "shape" &&
            state.role !== "parameter",
        )
        .slice(0, 200),
    [tensors, tensor.id, shape],
  );
  const target = other ? tensors[other] : null;
  const local = useMemo(
    () => (target ? closeness(tensor, target) : null),
    [tensor, target],
  );
  const [remote, setRemote] = useState<{
    key: string;
    summary: ChangeSummary | null;
  } | null>(null);
  const key = target && !local && flow.runId ? `${flow.runId}|${other}` : null;
  useEffect(() => {
    if (!key || !flow.runId) return;
    let current = true;
    loadComparison(flow.runId, flow.runId, [[tensor.id, other]])
      .then(([summary]) => current && setRemote({ key, summary }))
      .catch(() => current && setRemote({ key, summary: null }));
    return () => {
      current = false;
    };
  }, [key, flow.runId, tensor.id, other]);
  const summary = local ?? (remote?.key === key ? remote.summary : null);
  if (!candidates.length) return null;
  const label = (state: Tensor) => {
    const step = flow.uses(state.id).made?.step;
    return `${state.name} · ${step ? `step ${step}` : "input"}`;
  };
  return (
    <>
      <dt>Compare with</dt>
      <dd className="tensor-compare">
        <select
          aria-label={`Compare ${tensor.name} with`}
          value={other}
          onChange={(event) => {
            setOther(event.target.value);
            // A different choice replaces what the grid shows.
            if (shown) onShow(event.target.value || null);
          }}
        >
          <option value="">Another tensor of this shape…</option>
          {candidates.map((state) => (
            <option key={state.id} value={state.id}>
              {label(state)}
            </option>
          ))}
        </select>
        {target && (
          <span
            className={
              summary?.allclose
                ? "tensor-compare-same"
                : "tensor-compare-differs"
            }
            role="status"
          >
            {!summary
              ? key && remote?.key !== key
                ? "Comparing…"
                : "Could not compare these values."
              : summary.allclose
                ? `Close (torch.allclose) · ${summary.changed ? `max |Δ| ${formatValue(asNumber(summary.max_abs))}` : "identical"}`
                : `Differs: ${summary.changed.toLocaleString()} of ${summary.compared.toLocaleString()} values · max |Δ| ${formatValue(asNumber(summary.max_abs))} · mean |Δ| ${formatValue(asNumber(summary.mean_abs))}`}
          </span>
        )}
        {target && (
          <button
            type="button"
            className="text-button tensor-step-link"
            aria-pressed={shown === other}
            title="Shade every cell of the grid by its difference from this tensor"
            onClick={() => onShow(shown === other ? null : other)}
          >
            {shown === other
              ? "Stop showing in the grid"
              : "Show differences in the grid"}
          </button>
        )}
      </dd>
    </>
  );
}

/** A step this tensor came from or went to; selecting it opens that step. */
function StepLink({ step, go }: { step: UseStep; go: (id: string) => void }) {
  return (
    <button
      type="button"
      className="text-button tensor-step-link"
      title={`Open step ${step.step}, ${step.kind}`}
      onClick={() => go(step.id)}
    >
      {step.step} · {step.kind}
    </button>
  );
}

type CopyKind = "value" | "index" | "list" | "shape" | "slice" | "contract";

/** Copy the chosen value, or the whole tensor as a Python list when it fits. */
function CopyActions({
  value,
  tensor,
  coords,
  slice,
  runId,
}: {
  value: number | string | undefined;
  tensor: Tensor;
  coords: number[];
  /** The recorded run, for downloading the values as a .npy file. */
  runId?: string;
  /** The plane on screen, for tensors of more than two axes. */
  slice?: { name: string; build: () => string | null };
}) {
  const [copied, setCopied] = useState<CopyKind | null>(null);
  const list = tensor.numel <= 10_000 ? pythonList(tensor) : null;
  async function copy(kind: CopyKind, text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(kind);
      window.setTimeout(() => setCopied(null), 1400);
    } catch {
      setCopied(null);
    }
  }
  return (
    <div className="copy-actions">
      <button
        type="button"
        className="text-button"
        disabled={value === undefined}
        title={`Copy ${tensor.name}[${coords.join(", ")}] as it is stored`}
        onClick={() =>
          value !== undefined &&
          void copy("value", exactValue(value, tensor.dtype))
        }
      >
        {copied === "value" ? "Copied" : "Copy value"}
      </button>
      <button
        type="button"
        className="text-button"
        title={`Copy ${tensor.name}[${coords.join(", ")}], to read this element in code`}
        onClick={() =>
          void copy("index", `${tensor.name}[${coords.join(", ")}]`)
        }
      >
        {copied === "index" ? "Copied" : "Copy index"}
      </button>
      <button
        type="button"
        className="text-button"
        title="Copy the shape as torch.Size, for an assert or a reshape"
        onClick={() =>
          void copy("shape", `torch.Size([${tensor.shape.join(", ")}])`)
        }
      >
        {copied === "shape" ? "Copied" : "Copy shape"}
      </button>
      <button
        type="button"
        className="text-button"
        title={`Copy a contract stating this tensor's shape and range, to paste after its line: ${contractFor(tensor)}`}
        onClick={() => void copy("contract", contractFor(tensor))}
      >
        {copied === "contract" ? "Copied" : "Copy as contract"}
      </button>
      {slice && (
        <button
          type="button"
          className="text-button"
          title={`Copy ${slice.name}, the plane on screen, as a nested Python list`}
          onClick={() => {
            const text = slice.build();
            if (text) void copy("slice", text);
          }}
        >
          {copied === "slice" ? "Copied" : "Copy this slice"}
        </button>
      )}
      {list && (
        <button
          type="button"
          className="text-button"
          title={`Copy all ${tensor.numel.toLocaleString()} values as a nested Python list, ready for torch.tensor(...)`}
          onClick={() => void copy("list", list)}
        >
          {copied === "list" ? "Copied" : "Copy as Python list"}
        </button>
      )}
      {runId && (
        <a
          className="text-button"
          href={`/api/v1/runs/${runId}/tensors/${tensor.id}/npy`}
          download={`${tensor.name}.npy`}
          title={`Download all ${tensor.numel.toLocaleString()} values as a .npy file, for numpy.load or torch.from_numpy`}
        >
          Download .npy
        </a>
      )}
    </div>
  );
}

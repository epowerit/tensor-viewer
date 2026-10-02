import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RotateCcw,
} from "lucide-react";
import { formatCellValue } from "../tensors/coordinates";
import { useInspectionActive } from "../journey/InspectionActivity";
import { CoordinateJump } from "../tensors/TensorNavigation";
import { useTensorValues } from "../tensors/useTensorValues";
import { useCellPaint } from "../tensors/InkShape";
import "../tensors/gridInk.css";
import {
  CELL_SIZE,
  transitionPoint,
  transitionScene,
  transitionStep,
  type LayoutTransition,
  type Point,
  type transitionWindow,
} from "./layoutTransition";

type Props = {
  mapping: LayoutTransition;
  runId: string;
  showValues: boolean;
  onInputSelect: (index: number) => void;
  onOutputSelect: (index: number) => void;
};

function useReducedMotion() {
  const [reduced, setReduced] = useState(
    () => window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  return reduced;
}

const coord = (values: number[]) => `[${values.join(", ")}]`;

/** Replay the coordinate relationship, without mutating or re-executing tensors. */
export function LayoutTransitionView({
  mapping: m,
  runId,
  showValues,
  onInputSelect,
  onOutputSelect,
}: Props) {
  const active = useInspectionActive();
  const container = useRef<HTMLElement>(null);
  const [vertical, setVertical] = useState(false);
  const [progress, setProgress] = useState(0);
  const [playing, setPlaying] = useState(false);
  const progressRef = useRef(0);
  progressRef.current = progress;
  const reduced = useReducedMotion();
  const scene = useMemo(() => transitionScene(m, vertical), [m, vertical]);
  const position = transitionPoint(
    scene.start,
    scene.middle,
    scene.end,
    progress,
  );
  const shapeOnly =
    m.input.value_source === "shape" || m.output.value_source === "shape";
  const numeric = showValues && !shapeOnly;
  const inputData = useTensorValues(m.input, runId, numeric ? [m.source] : []);
  const outputData = useTensorValues(
    m.output,
    runId,
    numeric ? [m.target] : [],
  );
  const value =
    progress === 1 ? outputData.valueAt(m.target) : inputData.valueAt(m.source);
  const label = numeric ? formatCellValue(value, 4) : "●";
  const phase = progress < 0.25 ? 0 : progress < 0.75 ? 1 : 2;
  const error = inputData.error || outputData.error;

  useLayoutEffect(() => {
    setPlaying(false);
    setProgress(0);
  }, [m.source, m.target]);

  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width > 0)
        setVertical(entry.contentRect.width < 550);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!active || !playing || reduced) return;
    let frame = 0;
    let start: number | null = null;
    const from = progressRef.current;
    const tick = (time: number) => {
      if (start === null) start = time;
      const next = Math.min(1, from + (time - start) / 2400);
      setProgress(next);
      if (next < 1) frame = requestAnimationFrame(tick);
      else setPlaying(false);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [active, playing, reduced]);
  useEffect(() => {
    if (!active) setPlaying(false);
  }, [active]);
  useEffect(() => {
    const pause = () => {
      if (document.hidden) setPlaying(false);
    };
    document.addEventListener("visibilitychange", pause);
    return () => document.removeEventListener("visibilitychange", pause);
  }, []);
  useEffect(() => {
    if (reduced) setPlaying(false);
  }, [reduced]);

  function seek(value: number) {
    setPlaying(false);
    setProgress(value);
  }
  function play() {
    if (!active) return;
    if (playing) setPlaying(false);
    else {
      if (progress === 1) setProgress(0);
      setPlaying(true);
    }
  }
  // Cells are glass tinted by where each value came from; the followed one burns.
  const paint = useCellPaint();
  function drawWindow(
    window: ReturnType<typeof transitionWindow>,
    origin: Point,
    side: "Before" | "After",
  ) {
    const tensor = side === "Before" ? m.input : m.output;
    const index = side === "Before" ? m.source : m.target;
    const select = side === "Before" ? onInputSelect : onOutputSelect;
    const center = origin.x + (window.columns * CELL_SIZE) / 2;
    const captionY = vertical ? (side === "Before" ? 208 : 524) : 228;
    return (
      <g role="group" aria-label={`${side} mapping window`}>
        <text
          className="transition-window-title"
          x={center}
          y={vertical ? (side === "Before" ? 24 : 340) : 26}
          textAnchor="middle"
        >
          {side} ·{" "}
          {tensor.shape.length > 1
            ? "2D slice"
            : tensor.shape.length
              ? "vector"
              : "scalar"}
        </text>
        {Array.from({ length: window.columns }, (_, column) => (
          <text
            key={`column-${column}`}
            className="transition-axis"
            x={origin.x + (column + 0.5) * CELL_SIZE}
            y={origin.y - 10}
            textAnchor="middle"
            style={{
              fontSize: Math.min(
                10,
                40 / String(window.startColumn + column).length,
              ),
            }}
          >
            {window.startColumn + column}
          </text>
        ))}
        {Array.from({ length: window.rows }, (_, row) => (
          <text
            key={`row-${row}`}
            className="transition-axis"
            x={origin.x - 9}
            y={origin.y + (row + 0.5) * CELL_SIZE + 3}
            textAnchor="end"
            style={{
              fontSize: Math.min(10, 34 / String(window.startRow + row).length),
            }}
          >
            {window.startRow + row}
          </text>
        ))}
        {window.cells.map((cell) => (
          <g
            key={cell.index}
            role="button"
            tabIndex={0}
            aria-label={`${side} mapping cell ${coord(cell.coordinates)}`}
            aria-pressed={cell.index === index}
            onClick={() => {
              seek(0);
              select(cell.index);
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                seek(0);
                select(cell.index);
              }
            }}
          >
            <title>
              {coord(cell.coordinates)} · logical index {cell.index}
            </title>
            <rect
              className={`transition-cell ${cell.index === index ? "selected" : ""} ${cell.index !== index && paint?.(tensor, cell.index) ? "cell-inked" : ""}`}
              style={
                cell.index !== index && paint?.(tensor, cell.index)
                  ? ({
                      "--cell-ink": paint(tensor, cell.index),
                    } as React.CSSProperties)
                  : undefined
              }
              x={origin.x + cell.column * CELL_SIZE + 1}
              y={origin.y + cell.row * CELL_SIZE + 1}
              width={CELL_SIZE - 2}
              height={CELL_SIZE - 2}
              rx={4}
            />
          </g>
        ))}
        <text
          className="transition-axis"
          x={center}
          y={captionY}
          textAnchor="middle"
        >
          {window.cells.length} of {tensor.numel.toLocaleString()} cells
        </text>
        <text
          className="transition-axis"
          x={center}
          y={captionY + 16}
          textAnchor="middle"
        >
          {window.prefix.length
            ? `Fixed ${coord(window.prefix)}`
            : "All leading coordinates shown"}
        </text>
      </g>
    );
  }

  return (
    <section
      ref={container}
      className="layout-transition"
      aria-label="Element mapping replay"
    >
      <div className="transition-heading">
        <div>
          <h3>
            {m.rule === "permutation"
              ? "Reorder the coordinates"
              : JSON.stringify(m.input.shape) === JSON.stringify(m.output.shape)
                ? "Keep the coordinates, inspect storage"
                : "Regroup the same element"}
          </h3>
        </div>
        <span className="transition-scope">Logical mapping</span>
      </div>
      <div
        className="transition-stops"
        role="group"
        aria-label="Mapping explanation steps"
      >
        {[
          { title: "Original", detail: coord(m.before) },
          {
            title:
              m.rule === "permutation" ? "Reorder axes" : "Keep logical index",
            detail:
              m.rule === "permutation"
                ? coord(m.order!)
                : m.source.toLocaleString(),
          },
          { title: "Result", detail: coord(m.after) },
        ].map((step, index) => (
          <button
            key={step.title}
            aria-label={`Mapping step ${index + 1}: ${step.title}`}
            aria-pressed={phase === index}
            onClick={() => seek(index / 2)}
          >
            <span>
              <b>{index + 1}</b>
              {step.title}
            </span>
            <code>{step.detail}</code>
          </button>
        ))}
      </div>
      <div className="transition-playback">
        <button
          className="secondary-button"
          aria-label="Reset element animation"
          title="Back to original coordinate"
          onClick={() => seek(0)}
        >
          <RotateCcw size={14} />
        </button>
        <button
          className="secondary-button"
          aria-label="Previous mapping step"
          disabled={progress === 0}
          onClick={() => seek(transitionStep(progress, -1))}
        >
          <ChevronLeft size={15} />
        </button>
        <button
          className="primary-button"
          onClick={
            reduced
              ? () => seek(progress === 1 ? 0 : transitionStep(progress, 1))
              : play
          }
        >
          {playing ? <Pause size={14} /> : <Play size={14} />}
          {reduced
            ? "Advance"
            : playing
              ? "Pause"
              : progress === 1
                ? "Replay"
                : "Play mapping"}
        </button>
        <button
          className="secondary-button"
          aria-label="Next mapping step"
          disabled={progress === 1}
          onClick={() => seek(transitionStep(progress, 1))}
        >
          <ChevronRight size={15} />
        </button>
        <input
          type="range"
          min={0}
          max={100}
          step={1}
          value={Math.round(progress * 100)}
          aria-label="Element animation progress"
          aria-valuetext={`${Math.round(progress * 100)} percent, ${["original coordinate", "index mapping", "result coordinate"][phase]}`}
          onChange={(event) => seek(Number(event.target.value) / 100)}
        />
        <span className="transition-progress">
          {Math.round(progress * 100)}%
        </span>
      </div>
      <svg
        className="transition-canvas"
        viewBox={`0 0 ${scene.width} ${scene.height}`}
        role="group"
        aria-label="Animated coordinate mapping"
      >
        <title>
          Two bounded tensor windows. The moving marker follows one element's
          logical mapping, not a physical memory move.
        </title>
        {drawWindow(scene.before, scene.inputOrigin, "Before")}
        {drawWindow(scene.after, scene.outputOrigin, "After")}
        <path
          className="transition-path"
          d={`M ${scene.start.x} ${scene.start.y} L ${scene.middle.x} ${scene.middle.y} L ${scene.end.x} ${scene.end.y}`}
        />
        <text
          className="transition-rule-label"
          x={scene.middle.x}
          y={scene.middle.y - 30}
          textAnchor="middle"
        >
          {m.rule === "permutation" ? "axis order" : "logical index"}
        </text>
        <text
          className="transition-axis"
          x={scene.middle.x}
          y={scene.middle.y + 36}
          textAnchor="middle"
        >
          {m.rule === "permutation"
            ? coord(m.order!)
            : m.source.toLocaleString()}
        </text>
        <g
          className="transition-marker"
          transform={`translate(${position.x}, ${position.y})`}
          aria-hidden="true"
        >
          <rect x={-12} y={-12} width={24} height={24} rx={4} />
          <text textAnchor="middle" y={4}>
            {label.length > 4 ? "…" : label}
          </text>
        </g>
      </svg>
      <p className="transition-stage-text" role="status">
        {phase === 0
          ? `Start at input ${coord(m.before)}.`
          : phase === 1
            ? m.rule === "permutation"
              ? `Output axes read input axes in order ${coord(m.order!)}. Coordinates travel with their axes.`
              : `Logical index ${m.source.toLocaleString()} stays the same; decode it using output shape ${coord(m.output.shape)}.`
            : `Arrive at output ${coord(m.after)}. The value is unchanged.`}
      </p>
      <div className="transition-inspection">
        <div>
          <span>Input · shape {coord(m.input.shape)}</span>
          <code>{coord(m.before)}</code>
          {numeric && (
            <strong
              title={String(inputData.valueAt(m.source) ?? "unavailable")}
            >
              {inputData.valueAt(m.source) ?? "—"}
            </strong>
          )}
          <CoordinateJump
            label="Mapping input"
            shape={m.input.shape}
            coords={m.before}
            onSelect={onInputSelect}
          />
        </div>
        <div>
          <span>Output · shape {coord(m.output.shape)}</span>
          <code>{coord(m.after)}</code>
          {numeric && (
            <strong
              title={String(outputData.valueAt(m.target) ?? "unavailable")}
            >
              {outputData.valueAt(m.target) ?? "—"}
            </strong>
          )}
          <CoordinateJump
            label="Mapping output"
            shape={m.output.shape}
            coords={m.after}
            onSelect={onOutputSelect}
          />
        </div>
      </div>
      <p className="transition-footnote">
        {m.sharedStorage
          ? "Input and output share storage."
          : "The output uses separate storage."}{" "}
        {shapeOnly
          ? "Meta layout · no numeric values were recorded."
          : `Storage positions: ${m.sourceStorage.toLocaleString()} → ${m.targetStorage.toLocaleString()} (element offsets${m.sharedStorage ? " in the same storage" : " in different storages"}).`}{" "}
        The marker explains coordinates; it does not show a physical memory
        move.{" "}
        {reduced && "Reduced motion is on; use the steps or progress control."}
      </p>
      {numeric && (inputData.loading || outputData.loading) && (
        <p className="transition-footnote" role="status">
          Loading the selected recorded values…
        </p>
      )}
      {numeric && error && (
        <p className="transition-error" role="alert">
          {error}{" "}
          <button
            className="text-button"
            onClick={() => {
              inputData.retry();
              outputData.retry();
            }}
          >
            Retry values
          </button>
        </p>
      )}
    </section>
  );
}

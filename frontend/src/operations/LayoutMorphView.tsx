import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { formatCellValue, unravel } from "../tensors/coordinates";
import { useTensorValues } from "../tensors/useTensorValues";
import {
  MORPH_CELL,
  morphColor,
  morphPosition,
  morphScene,
  type CellLayout,
  type LayoutMorph,
  type Point,
} from "./layoutMorph";
import { usePlayback } from "./usePlayback";

type Props = {
  morph: LayoutMorph;
  runId: string;
  showValues: boolean;
  /** Selected output cell. */
  selected: number;
  onSelect: (target: number) => void;
};

const coord = (values: number[]) => `[${values.join(", ")}]`;
/** Axis names when they are meaningful and fit above the grid. */
function describe(tensor: LayoutMorph["input"], width: number) {
  const named = tensor.axes
    .map((axis, i) => `${axis} ${tensor.shape[i]}`)
    .join(" × ");
  return tensor.axes.some((axis) => !/^axis \d+$/.test(axis)) &&
    named.length * 6.7 <= width
    ? named
    : coord(tensor.shape);
}

/** Every recorded cell travels from its input position to its output position. */
export function LayoutMorphView({
  morph: m,
  runId,
  showValues,
  selected,
  onSelect,
}: Props) {
  const container = useRef<HTMLElement>(null);
  const [narrow, setNarrow] = useState(false);
  const scene = useMemo(() => morphScene(m, narrow), [m, narrow]);
  const count = m.movers.length;
  const playback = usePlayback(
    1400 + Math.min(count, 96) * 18,
    `${runId}/${m.input.id}/${m.output.id}`,
    true,
  );
  const shapeOnly = m.input.value_source === "shape";
  const labelled = showValues && !shapeOnly && m.input.numel <= 96;
  const inputIndices = useMemo(
    () => (labelled ? [...Array(m.input.numel).keys()] : []),
    [labelled, m.input.numel],
  );
  const outputIndices = useMemo(
    () =>
      labelled && m.kind === "collapse"
        ? [...Array(m.output.numel).keys()]
        : [],
    [labelled, m.kind, m.output.numel],
  );
  const values = useTensorValues(m.input, runId, inputIndices);
  const results = useTensorValues(m.output, runId, outputIndices);
  const target = selected < m.output.numel ? selected : 0;
  const chosen = m.movers.filter((mover) => mover.target === target);
  const landed = playback.progress >= 1;

  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) =>
      setNarrow(entry.contentRect.width < 520),
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  function slots(layout: CellLayout, origin: Point) {
    return layout.points.map((point, i) => (
      <rect
        key={i}
        className="morph-slot"
        x={origin.x + point.x + 1}
        y={origin.y + point.y + 1}
        width={MORPH_CELL - 2}
        height={MORPH_CELL - 2}
        rx={3}
      />
    ));
  }
  function cell(
    key: string,
    position: Point,
    source: number,
    label: number | string | undefined,
    className: string,
    select?: () => void,
  ) {
    return (
      <g
        key={key}
        className={`morph-cell ${className}`}
        transform={`translate(${position.x}, ${position.y})`}
        onClick={select}
      >
        <rect
          x={1}
          y={1}
          width={MORPH_CELL - 2}
          height={MORPH_CELL - 2}
          rx={3}
          fill={morphColor(source, m.input.numel)}
        />
        {labelled && label !== undefined && (
          <text x={MORPH_CELL / 2} y={MORPH_CELL / 2 + 3.5}>
            {formatCellValue(label, 3)}
          </text>
        )}
      </g>
    );
  }
  // Raise the selected cells above their neighbors.
  const order = m.movers
    .map((_, i) => i)
    .sort(
      (a, b) =>
        Number(m.movers[a].target === target) -
        Number(m.movers[b].target === target),
    );

  return (
    <section
      ref={container}
      className="layout-morph"
      aria-label="Whole-tensor motion"
    >
      <div className="morph-playback">
        <button
          className="icon-button"
          aria-label="Show the input arrangement"
          disabled={playback.progress === 0}
          onClick={() => playback.seek(0)}
        >
          <SkipBack size={14} />
        </button>
        <button
          className="primary-button small"
          disabled={playback.reduced}
          onClick={playback.toggle}
        >
          {playback.playing ? <Pause size={13} /> : <Play size={13} />}
          {playback.playing
            ? "Pause"
            : playback.progress >= 1
              ? "Replay"
              : "Play"}
        </button>
        <button
          className="icon-button"
          aria-label="Show the output arrangement"
          disabled={playback.progress === 1}
          onClick={() => playback.seek(1)}
        >
          <SkipForward size={14} />
        </button>
        <input
          type="range"
          min={0}
          max={100}
          value={Math.round(playback.progress * 100)}
          aria-label="Motion progress"
          aria-valuetext={`${Math.round(playback.progress * 100)} percent`}
          onChange={(event) => playback.seek(Number(event.target.value) / 100)}
        />
      </div>
      <svg
        className="morph-canvas"
        viewBox={`0 0 ${scene.width} ${scene.height}`}
        style={{ maxWidth: scene.width * 1.35 }}
        role="group"
        aria-label={`${count} cells move from shape ${coord(m.input.shape)} to shape ${coord(m.output.shape)}`}
      >
        <text
          className="morph-title"
          x={scene.inputOrigin.x}
          y={scene.inputOrigin.y - 12}
        >
          Before · {describe(m.input, m.before.width - 50)}
        </text>
        <text
          className="morph-title"
          x={scene.outputOrigin.x}
          y={scene.outputOrigin.y - 12}
        >
          After · {describe(m.output, m.after.width - 44)}
        </text>
        {slots(m.before, scene.inputOrigin)}
        {slots(m.after, scene.outputOrigin)}
        {m.kind === "select" &&
          m.before.points.map((point, source) =>
            cell(
              `stay-${source}`,
              {
                x: scene.inputOrigin.x + point.x,
                y: scene.inputOrigin.y + point.y,
              },
              source,
              values.valueAt(source),
              "resting",
            ),
          )}
        {order.map((i) => {
          const mover = m.movers[i];
          return cell(
            `move-${i}`,
            morphPosition(m, scene, i, playback.progress),
            mover.source,
            values.valueAt(mover.source),
            mover.target === target ? "selected" : "",
            () => onSelect(mover.target),
          );
        })}
        {m.kind === "collapse" &&
          landed &&
          m.after.points.map((point, index) => (
            <g
              key={`result-${index}`}
              className={`morph-cell morph-result ${index === target ? "selected" : ""}`}
              transform={`translate(${scene.outputOrigin.x + point.x}, ${scene.outputOrigin.y + point.y})`}
              onClick={() => onSelect(index)}
            >
              <rect
                x={1}
                y={1}
                width={MORPH_CELL - 2}
                height={MORPH_CELL - 2}
                rx={3}
              />
              {labelled && results.valueAt(index) !== undefined && (
                <text x={MORPH_CELL / 2} y={MORPH_CELL / 2 + 3.5}>
                  {formatCellValue(results.valueAt(index), 3)}
                </text>
              )}
            </g>
          ))}
      </svg>
      <p className="morph-readout" role="status">
        <span>
          Selected:{" "}
          {chosen.length === 1 ? (
            <>
              input{" "}
              <code>{coord(unravel(chosen[0].source, m.input.shape))}</code>
            </>
          ) : (
            `${chosen.length} input cells`
          )}{" "}
          → output <code>{coord(unravel(target, m.output.shape))}</code>
        </span>
        <span>
          {m.caption} Colors follow input order.
          {playback.reduced &&
            " Reduced motion is on; use the slider or the skip buttons."}
        </span>
      </p>
      {(values.error || results.error) && (
        <p className="transition-error" role="alert">
          {values.error || results.error}{" "}
          <button
            className="text-button"
            onClick={() => {
              values.retry();
              results.retry();
            }}
          >
            Retry values
          </button>
        </p>
      )}
    </section>
  );
}

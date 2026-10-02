import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { formatCellValue, unravel } from "../tensors/coordinates";
import { inkShade, storyAxis } from "../tensors/inkShade";
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
import type { Ink } from "../tensors/axisInk";
import { useAxisInk } from "../tensors/InkShape";

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
/** The shape in an SVG title: axis names when they fit, each size inked. */
function Describe({
  tensor,
  width,
  ink,
}: {
  tensor: LayoutMorph["input"];
  width: number;
  ink: (Ink | null)[] | null;
}) {
  const named = tensor.axes
    .map((axis, i) => `${axis} ${tensor.shape[i]}`)
    .join(" × ");
  const useNames =
    tensor.axes.some((axis) => !/^axis \d+$/.test(axis)) &&
    named.length * 6.7 <= width;
  if (!tensor.shape.length) return <>{coord(tensor.shape)}</>;
  const sizes = tensor.shape.map((size, axis) => (
    <Fragment key={axis}>
      {axis > 0 && (useNames ? " × " : ", ")}
      {useNames && `${tensor.axes[axis]} `}
      <tspan
        className={ink?.[axis] ? "morph-ink" : undefined}
        fill={ink?.[axis]?.colors[0]}
      >
        {size}
        {ink?.[axis] && <title>{ink[axis]!.text}</title>}
      </tspan>
    </Fragment>
  ));
  return useNames ? <>{sizes}</> : <>[{sizes}]</>;
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
  const inputInk = useAxisInk(m.input);
  const outputInk = useAxisInk(m.output);
  const [narrow, setNarrow] = useState(false);
  const scene = useMemo(() => morphScene(m, narrow), [m, narrow]);
  // Cells take the ink of one input axis, shaded by their position along it,
  // so the colors that name an axis in shapes are the colors that move here.
  const suggested = useMemo(
    () => storyAxis(m.input.shape, inputInk, outputInk),
    [m.input.shape, inputInk, outputInk],
  );
  const [picked, setPicked] = useState<number | "order" | null>(null);
  const colorBy =
    picked === "order" || (picked !== null && inputInk?.[picked])
      ? picked
      : (suggested ?? "order");
  const axisLabel = (axis: number) =>
    /^axis \d+$/.test(m.input.axes[axis] ?? "")
      ? `axis ${axis}`
      : m.input.axes[axis];
  function paint(source: number) {
    if (colorBy === "order")
      return { fill: morphColor(source, m.input.numel), text: undefined };
    const position = unravel(source, m.input.shape)[colorBy];
    return inkShade(inputInk![colorBy]!, position, m.input.shape[colorBy]);
  }
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
    // The selected cell burns, whatever the coloring.
    const color = className.includes("selected")
      ? { fill: "url(#tv-fire)", text: "#3b1406" }
      : paint(source);
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
          fill={color.fill}
        />
        {labelled && label !== undefined && (
          <text
            x={MORPH_CELL / 2}
            y={MORPH_CELL / 2 + 3.5}
            style={color.text ? { fill: color.text } : undefined}
          >
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
      {suggested !== null && (
        <div
          className="morph-color-by"
          role="radiogroup"
          aria-label="Color cells by"
        >
          <span>Color by</span>
          {m.input.shape.map((size, axis) => {
            const ink = inputInk?.[axis];
            if (!ink || size < 2) return null;
            return (
              <button
                key={axis}
                role="radio"
                aria-checked={colorBy === axis}
                title={`Shade each cell by its position along ${ink.text}`}
                onClick={() => setPicked(axis)}
              >
                <span
                  className="morph-swatch"
                  style={{
                    background: `linear-gradient(90deg, ${inkShade(ink, 0, size).fill}, ${inkShade(ink, size - 1, size).fill})`,
                  }}
                />
                {axisLabel(axis)} <b>{size}</b>
              </button>
            );
          })}
          <button
            role="radio"
            aria-checked={colorBy === "order"}
            title="Color cells by their order in memory"
            onClick={() => setPicked("order")}
          >
            <span
              className="morph-swatch"
              style={{
                background: `linear-gradient(90deg, ${morphColor(0, 2)}, ${morphColor(1, 2)})`,
              }}
            />
            order
          </button>
        </div>
      )}
      <svg
        className="morph-canvas"
        viewBox={`0 0 ${scene.width} ${scene.height}`}
        style={{ maxWidth: scene.width * 2.4 }}
        role="group"
        aria-label={`${count} cells move from shape ${coord(m.input.shape)} to shape ${coord(m.output.shape)}`}
      >
        <text
          className="morph-title"
          x={scene.inputOrigin.x}
          y={scene.inputOrigin.y - 12}
        >
          Before ·{" "}
          <Describe
            tensor={m.input}
            width={m.before.width - 50}
            ink={inputInk}
          />
        </text>
        <text
          className="morph-title"
          x={scene.outputOrigin.x}
          y={scene.outputOrigin.y - 12}
        >
          After ·{" "}
          <Describe
            tensor={m.output}
            width={m.after.width - 44}
            ink={outputInk}
          />
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
          {m.caption}{" "}
          {colorBy === "order"
            ? "Colors follow input order."
            : `Shades follow ${inputInk![colorBy]!.text}: darkest at position 0, lightest at ${m.input.shape[colorBy] - 1}.`}
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

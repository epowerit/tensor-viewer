import type { Tensor } from "../api/client";
import type { AxisPart } from "../tensors/axisLineage";
import { INK_PALETTE, NEUTRAL_INK } from "../tensors/axisInk";
import { useAxisInk } from "../tensors/InkShape";

/** What a layout capsule did to its tensor's axes, for its wiring diagram. */
export type AxisWiringData = {
  input: Tensor;
  /** Each input axis's name, or null when the code gave none. */
  fromNames: (string | null)[];
  to: number[];
  toNames: (string | null)[];
  /** For each result axis, the input axes or pieces it is made of. */
  map: AxisPart[][];
};

const SLOT = 64,
  CHIP = 52,
  TOP = 4,
  CHIP_HEIGHT = 26,
  GAP = 42;

/**
 * A layout capsule's axes as wiring: the input's axes above, the result's
 * below, and a line for every part that travels. One axis cut in pieces
 * fans out, axes merged fan in, axes reordered cross; each input axis keeps
 * its axis ink, so `tokens` is the same color here as in every shape. Read
 * from the trace, the same drawing explains a head split, a window cut, or a
 * patchify in any model.
 */
export function AxisWiring({ data }: { data: AxisWiringData }) {
  const ink = useAxisInk(data.input);
  const { input, map, to } = data;
  const from = input.shape;
  const columns = Math.max(from.length, to.length, 1);
  const width = columns * SLOT;
  const height = TOP + CHIP_HEIGHT * 2 + GAP + 2;
  const x = (count: number, index: number) =>
    (width - count * SLOT) / 2 + index * SLOT + SLOT / 2;
  // An input axis's own ink, or a distinct color where it has only neutral ink.
  const color = (axis: number) => {
    const own = ink?.[axis]?.colors[0];
    return own && own !== NEUTRAL_INK
      ? own
      : INK_PALETTE[axis % INK_PALETTE.length];
  };
  // Where each line leaves its input chip and enters its result chip: pieces
  // of one axis leave side by side, parts of one result arrive side by side.
  const spread = (count: number, index: number) =>
    count > 1 ? ((index + 0.5) / count - 0.5) * (CHIP - 14) : 0;
  const leaving = new Map<number, number>();
  const pieces = (axis: number) =>
    map.flat().filter((part) => part.axis === axis).length;
  const lines = map.flatMap((parts, out) =>
    parts.map((part, index) => {
      const order = leaving.get(part.axis) ?? 0;
      leaving.set(part.axis, order + 1);
      return {
        key: `${out}-${index}`,
        color: color(part.axis),
        x1: x(from.length, part.axis) + spread(pieces(part.axis), order),
        x2: x(to.length, out) + spread(parts.length, index),
        piece: part.piece,
      };
    }),
  );
  const y1 = TOP + CHIP_HEIGHT,
    y2 = TOP + CHIP_HEIGHT + GAP;
  const chip = (
    key: string,
    cx: number,
    y: number,
    size: number,
    name: string | null,
    colors: string[],
  ) => (
    <g key={key} className="axis-wiring-chip">
      <rect
        x={cx - CHIP / 2}
        y={y}
        width={CHIP}
        height={CHIP_HEIGHT}
        rx={6}
        style={{ stroke: colors[0] ?? NEUTRAL_INK }}
      />
      {colors.length > 1 &&
        colors
          .slice(1)
          .map((stroke, i) => (
            <line
              key={i}
              x1={cx - CHIP / 2 + 6 + i * 5}
              x2={cx - CHIP / 2 + 6 + i * 5}
              y1={y + 4}
              y2={y + CHIP_HEIGHT - 4}
              style={{ stroke }}
            />
          ))}
      {name && (
        <text className="axis-wiring-name" x={cx} y={y + 10}>
          {name.length > 9 ? `${name.slice(0, 8)}…` : name}
        </text>
      )}
      <text className="axis-wiring-size" x={cx} y={y + (name ? 21 : 17)}>
        {size}
      </text>
    </g>
  );
  const describe = (sizes: number[], names: (string | null)[]) =>
    sizes.map((size, i) => (names[i] ? `${names[i]} ${size}` : `${size}`));
  return (
    <svg
      className="axis-wiring"
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      role="img"
      aria-label={`Axes [${describe(from, data.fromNames).join(", ")}] become [${describe(to, data.toNames).join(", ")}]`}
    >
      {lines.map((line) => (
        <path
          key={line.key}
          className={`axis-wiring-line${line.piece ? " is-piece" : ""}`}
          d={`M${line.x1},${y1} C${line.x1},${y1 + GAP / 2} ${line.x2},${y2 - GAP / 2} ${line.x2},${y2}`}
          style={{ stroke: line.color }}
        />
      ))}
      {from.map((size, axis) =>
        chip(
          `in-${axis}`,
          x(from.length, axis),
          TOP,
          size,
          data.fromNames[axis],
          [color(axis)],
        ),
      )}
      {to.map((size, out) =>
        chip(
          `out-${out}`,
          x(to.length, out),
          y2,
          size,
          data.toNames[out],
          map[out]?.length
            ? map[out].map((part) => color(part.axis))
            : [NEUTRAL_INK],
        ),
      )}
    </svg>
  );
}

import type { Ink } from "./axisInk";

/** HSL components of a #rrggbb color. */
export function hexToHsl(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map(
    (c) => c / 255,
  );
  const max = Math.max(r, g, b),
    min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l * 100];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  const h =
    max === r
      ? ((g - b) / d + (g < b ? 6 : 0)) * 60
      : max === g
        ? ((b - r) / d + 2) * 60
        : ((r - g) / d + 4) * 60;
  return [h, s * 100, l * 100];
}

/**
 * The color of a cell at `position` along an inked axis of `size` positions:
 * the axis ink, from dark at the first position to light at the last, so the
 * same hue that marks the axis in shapes marks its cells while they move. A
 * merged axis walks through its sources' inks in order.
 */
export function inkShade(
  ink: Ink,
  position: number,
  size: number,
): { fill: string; text: string } {
  const t = size > 1 ? Math.min(Math.max(position / (size - 1), 0), 1) : 0.5;
  const segment = Math.min(
    ink.colors.length - 1,
    Math.floor(t * ink.colors.length),
  );
  // Position within the segment, so each source's ink keeps its own ramp.
  const local = ink.colors.length > 1 ? t * ink.colors.length - segment : t;
  const [h, s] = hexToHsl(ink.colors[segment]);
  const lightness = 30 + 36 * local;
  return {
    fill: `hsl(${Math.round(h)} ${Math.round(Math.min(s, 72))}% ${Math.round(lightness)}%)`,
    // Light shades take dark labels so values stay readable.
    text: lightness > 54 ? "#1b1525" : "#ffffff",
  };
}

/**
 * The input axis whose shades best show what a step does: one that moves to
 * another position (a transpose), else one that is split, merged, or reduced,
 * else the last inked axis. Larger axes win ties. Null when nothing is inked.
 */
export function storyAxis(
  shape: number[],
  inputInk: (Ink | null)[] | null,
  outputInk: (Ink | null)[] | null,
): number | null {
  const candidates = shape.flatMap((size, axis) =>
    size > 1 && inputInk?.[axis] ? [axis] : [],
  );
  if (!candidates.length) return null;
  const at = (axis: number) =>
    outputInk?.findIndex((ink) => ink?.text === inputInk![axis]!.text) ?? -1;
  const best = (axes: number[]) =>
    axes.reduce((a, b) => (shape[b] >= shape[a] ? b : a));
  const moved = candidates.filter((axis) => at(axis) >= 0 && at(axis) !== axis);
  if (moved.length) return best(moved);
  const changed = candidates.filter((axis) => at(axis) < 0);
  if (changed.length) return best(changed);
  return candidates.at(-1)!;
}

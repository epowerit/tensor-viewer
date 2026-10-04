import type { AxisPart } from "../tensors/axisLineage";

/**
 * A rearrangement as einops writes it: `b n (h d) -> b h n d`, with the
 * sizes einops needs to split an axis (`h=2`). Read from an axis map, so it
 * says what the reshapes and permutes did, whichever calls the code used.
 */
export type Einops = {
  pattern: string;
  sizes: [string, number][];
  /** `rearrange(x, 'b n (h d) -> b h n d', h=2)` */
  call: (variable: string) => string;
};

/** An einops axis name from an axis name: a short identifier. */
function identifier(name: string | null, fallback: string): string {
  const cleaned = (name ?? "")
    .toLowerCase()
    .replace(/[^a-z0-9_]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return /^[a-z_]/.test(cleaned) ? cleaned : fallback;
}

/**
 * The einops pattern of a pure rearrangement from an input of `from` sizes
 * to a result of `to` sizes, given where each result axis comes from. Null
 * when the map drops data or is not a rearrangement einops can write.
 */
export function einopsPattern(
  from: number[],
  fromNames: (string | null)[],
  to: number[],
  toNames: (string | null)[],
  map: AxisPart[][],
): Einops | null {
  if (map.length !== to.length) return null;
  const used = new Set<string>();
  const unique = (name: string) => {
    let candidate = name;
    for (let n = 2; used.has(candidate); n++) candidate = `${name}${n}`;
    used.add(candidate);
    return candidate;
  };
  // Where each input axis goes: whole, or as pieces.
  const whole = new Map<number, string>();
  const pieces = new Map<number, Map<number, string>>();
  const sizes: [string, number][] = [];
  const base = from.map((_, axis) => identifier(fromNames[axis], `a${axis}`));
  for (const [out, parts] of map.entries())
    for (const part of parts) {
      if (part.axis < 0 || part.axis >= from.length) return null;
      if (!part.piece) {
        if (!whole.has(part.axis))
          whole.set(part.axis, unique(base[part.axis]));
        continue;
      }
      const named = pieces.get(part.axis) ?? new Map<number, string>();
      if (!named.has(part.piece.index)) {
        // A piece standing alone in a named result axis takes that name.
        const own = parts.length === 1 ? identifier(toNames[out], "") : "";
        const symbol = unique(own || `${base[part.axis]}${part.piece.index}`);
        named.set(part.piece.index, symbol);
        sizes.push([symbol, part.size]);
      }
      pieces.set(part.axis, named);
    }
  const left: string[] = [];
  for (const [axis, size] of from.entries()) {
    const split = pieces.get(axis);
    if (split) {
      if (whole.has(axis)) return null;
      const of = map.flat().find((part) => part.axis === axis)?.piece?.of ?? 0;
      if (split.size !== of) return null;
      left.push(
        `(${[...split.entries()]
          .sort(([a], [b]) => a - b)
          .map(([, symbol]) => symbol)
          .join(" ")})`,
      );
    } else if (whole.has(axis)) left.push(whole.get(axis)!);
    // An axis of size 1 may be dropped; any other would lose values.
    else if (size === 1) left.push("1");
    else return null;
  }
  const symbolOf = (part: AxisPart) =>
    part.piece
      ? pieces.get(part.axis)!.get(part.piece.index)!
      : whole.get(part.axis)!;
  const right = map.map((parts, out) =>
    !parts.length
      ? to[out] === 1
        ? "1"
        : null
      : parts.length === 1
        ? symbolOf(parts[0])
        : `(${parts.map(symbolOf).join(" ")})`,
  );
  if (right.some((term) => term === null)) return null;
  const pattern = `${left.join(" ")} -> ${right.join(" ")}`;
  // einops infers the last piece of each split; the rest it is told.
  const told = sizes.filter(([symbol]) => {
    for (const split of pieces.values()) {
      const symbols = [...split.entries()].sort(([a], [b]) => a - b);
      if (symbols.at(-1)?.[1] === symbol) return false;
    }
    return true;
  });
  return {
    pattern,
    sizes: told,
    call: (variable) =>
      `rearrange(${variable}, '${pattern}'${told.map(([symbol, size]) => `, ${symbol}=${size}`).join("")})`,
  };
}

/** A tensor's name as a Python variable: `x`, `scores`, `x_1`. */
export const variableName = (name: string) =>
  identifier(name.replace(/[^A-Za-z0-9_]+/g, "_"), "x");

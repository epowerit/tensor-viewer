import type { CSSProperties } from "react";

/**
 * What a table's or map's shading means: a bar from no colour to full, with
 * the values at each end, in the colour the cells use.
 */
export function ShadeScale({
  color,
  low = "0%",
  high = "100%",
  strength = 1,
}: {
  color: string;
  low?: string;
  high?: string;
  /** How strong the fullest cell is drawn, from 0 to 1. */
  strength?: number;
}) {
  return (
    <span
      className="shade-scale"
      style={
        {
          "--shade": color,
          "--shade-full": `${Math.round(strength * 100)}%`,
        } as CSSProperties
      }
    >
      {low} <span aria-hidden="true" /> {high}
    </span>
  );
}

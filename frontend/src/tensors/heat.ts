import { useSyncExternalStore } from "react";
import type { Tensor } from "../api/client";

export const HEAT_COLD = "#4b93e0";
export const HEAT_WARM = "#dc5c7e";

/**
 * How strongly a value shades its cell, from -1 (most negative) to 1 (most
 * positive). A range spanning zero is diverging around zero; a range on one
 * side of zero is sequential from its smallest magnitude to its largest.
 */
export function heatLevel(value: number, low: number, high: number) {
  if (
    !Number.isFinite(value) ||
    !Number.isFinite(low) ||
    !Number.isFinite(high)
  )
    return null;
  if (low < 0 && high > 0) {
    const reach = Math.max(-low, high);
    return Math.max(-1, Math.min(1, value / reach));
  }
  if (high <= 0) {
    // All negative: the most negative value is the coldest.
    const span = high - low;
    return span ? -Math.max(0, Math.min(1, (high - value) / span)) : -1;
  }
  const span = high - low;
  return span ? Math.max(0, Math.min(1, (value - low) / span)) : 1;
}

/** The whole tensor's finite range, which heat shading spans; null without. */
export function heatRangeOf(
  tensor: Pick<Tensor, "minimum" | "maximum" | "value_source">,
) {
  const { minimum: low, maximum: high } = tensor;
  return tensor.value_source !== "shape" &&
    typeof low === "number" &&
    typeof high === "number" &&
    Number.isFinite(low) &&
    Number.isFinite(high)
    ? { low, high }
    : null;
}

/** A cell fill for a heat level, mixed into the grid's own base color. */
export function heatFill(level: number) {
  const share = Math.round(8 + Math.abs(level) * 72);
  return `color-mix(in srgb, ${level < 0 ? HEAT_COLD : HEAT_WARM} ${share}%, var(--tensor-base, #221d2d))`;
}

/** The legend's gradient for a range, matching `heatFill`. */
export function heatGradient(low: number, high: number) {
  const base = "var(--tensor-base, #221d2d)";
  if (low < 0 && high > 0) {
    const reach = Math.max(-low, high);
    const zero = (-low / (high - low)) * 100;
    return `linear-gradient(90deg, ${heatFill(low / reach)}, ${base} ${zero}%, ${heatFill(high / reach)})`;
  }
  return high <= 0
    ? `linear-gradient(90deg, ${heatFill(-1)}, ${heatFill(-0.01)})`
    : `linear-gradient(90deg, ${heatFill(0.01)}, ${heatFill(1)})`;
}

/**
 * A view switch shared by every grid, so a step's operands and result always
 * read on the same terms. With a storage key it is remembered across visits.
 */
function sharedToggle(event: string, key: string | null) {
  // The choice in this page, kept when storage is unavailable.
  let memory = false;
  const read = () => {
    if (!key) return memory;
    try {
      const stored = localStorage.getItem(key);
      // "heat" is how the first version stored the heat switch.
      return stored === null ? memory : stored === "on" || stored === "heat";
    } catch {
      return memory;
    }
  };
  const subscribe = (changed: () => void) => {
    window.addEventListener(event, changed);
    window.addEventListener("storage", changed);
    return () => {
      window.removeEventListener(event, changed);
      window.removeEventListener("storage", changed);
    };
  };
  const set = (next: boolean) => {
    memory = next;
    if (key)
      try {
        localStorage.setItem(key, next ? "on" : "off");
      } catch {
        // Without storage the choice still applies until the page reloads.
      }
    window.dispatchEvent(new Event(event));
  };
  return function useToggle(): [boolean, (on: boolean) => void] {
    return [useSyncExternalStore(subscribe, read, () => false), set];
  };
}

/** Whether grids shade cells by value, like a heatmap. Remembered. */
export const useHeat = sharedToggle(
  "tensorviewer-cell-shading",
  "tensorviewer.cellShading",
);

/** Whether grids show the change since the run before. For this visit. */
export const useDiff = sharedToggle("tensorviewer-cell-diff", null);

/** Whether grids show a 16 × 16 window of a large plane instead of 8 × 8. Remembered. */
export const useWideWindow = sharedToggle(
  "tensorviewer-grid-window",
  "tensorviewer.gridWindow",
);

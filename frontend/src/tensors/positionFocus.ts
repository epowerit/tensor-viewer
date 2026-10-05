import { useEffect, useRef, useSyncExternalStore } from "react";

/**
 * The position (word) under the pointer in any of the panels that read a
 * sentence position by position (the logit lens, causal trace, attention,
 * and map), so each of them can mark the same one.
 */
let focused: number | null = null;
const listeners = new Set<() => void>();

export function focusPosition(position: number | null) {
  if (position === focused) return;
  focused = position;
  listeners.forEach((listener) => listener());
}

export function usePositionFocus(): number | null {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => focused,
    () => null,
  );
}

/** Pointer handlers that focus a position while it is hovered. */
export const hoverPosition = (position: number) => ({
  onMouseEnter: () => focusPosition(position),
  onMouseLeave: () => focusPosition(null),
});

/**
 * A table whose columns are positions: when a position is focused, from this
 * panel or another, its header scrolls into view. Put the returned ref on the
 * table and `data-position` on each header cell.
 */
export function useFocusedColumn<T extends HTMLElement>(
  focused: number | null,
) {
  const table = useRef<T>(null);
  useEffect(() => {
    if (focused === null) return;
    table.current
      ?.querySelector(`[data-position="${focused}"]`)
      ?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [focused]);
  return table;
}

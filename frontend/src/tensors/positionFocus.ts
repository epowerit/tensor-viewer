import { useEffect, useRef, useSyncExternalStore } from "react";

/**
 * The position (word) in focus in any of the panels that read a sentence
 * position by position (the logit lens, causal trace, attention, and map), so
 * each of them can mark the same one: the word under the pointer, or else the
 * word locked by a click, which stays in focus while the pointer moves to
 * another panel.
 */
let hovered: number | null = null;
let locked: { position: number; word: string } | null = null;
const listeners = new Set<() => void>();
const changed = () => listeners.forEach((listener) => listener());
const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
};

export function focusPosition(position: number | null) {
  if (position === hovered) return;
  hovered = position;
  changed();
}

const unlockOnEscape = (event: KeyboardEvent) => {
  if (event.key === "Escape" && !event.defaultPrevented) lockPosition(null);
};

/** Locks a position in focus, or unlocks it when it is the locked one. */
export function lockPosition(position: number | null, word = "") {
  const next =
    position === null || position === locked?.position
      ? null
      : { position, word };
  if (next === locked) return;
  locked = next;
  if (typeof document !== "undefined") {
    if (locked) document.addEventListener("keydown", unlockOnEscape);
    else document.removeEventListener("keydown", unlockOnEscape);
  }
  changed();
}

/** The hovered position, or else the locked one. */
export const focusedPosition = () => hovered ?? locked?.position ?? null;

export function usePositionFocus(): number | null {
  return useSyncExternalStore(subscribe, focusedPosition, () => null);
}

/** The locked position and its word, if one is locked. */
export const lockedPosition = () => locked;

export function useLockedPosition() {
  return useSyncExternalStore(subscribe, lockedPosition, () => null);
}

/**
 * Pointer handlers that focus a position while it is hovered, and lock it
 * on a click.
 */
export const hoverPosition = (position: number, word = `${position}`) => ({
  "data-focusable": "",
  onMouseEnter: () => focusPosition(position),
  onMouseLeave: () => focusPosition(null),
  onClick: () => lockPosition(position, word),
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

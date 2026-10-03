import { useState } from "react";

/** How far past the smallest size a drag goes before it closes the panel. */
export const COLLAPSE = 72;

/**
 * A panel's edge: drag it to resize the panel, or focus it and use the arrow
 * keys (Shift for bigger steps), Home and End for the smallest and largest,
 * and Enter for the usual size, as a double-click does. Dragging well past
 * the smallest size closes the panel, which reopens at the size it had.
 *
 * `grow` is the direction that enlarges the panel: a side bar on the left
 * grows rightward, a column on the right leftward, and a panel along the
 * bottom upward. `measure` gives the panel's drawn size when it has no size
 * of its own yet, so a drag starts where the panel is.
 */
export function EdgeResizer({
  className,
  label,
  grow,
  size,
  min,
  max,
  onSize,
  onReset,
  onCollapse,
  measure,
}: {
  className: string;
  label: string;
  grow: "right" | "left" | "up";
  size: number;
  min: number;
  max: number;
  onSize: (size: number) => void;
  onReset: () => void;
  onCollapse?: () => void;
  measure?: () => number;
}) {
  const [drag, setDrag] = useState<"resize" | "collapse" | null>(null);
  const across = grow !== "up";
  // Moving the pointer this way enlarges the panel.
  const sign = grow === "right" ? 1 : -1;
  const current = () => Math.round(measure?.() || size);
  const bodyClass = across ? "resizing-across" : "resizing-down";
  const [less, more] =
    grow === "right"
      ? ["ArrowLeft", "ArrowRight"]
      : grow === "left"
        ? ["ArrowRight", "ArrowLeft"]
        : ["ArrowDown", "ArrowUp"];
  return (
    <div
      className={`edge-resizer ${className}`}
      data-drag={drag ?? undefined}
      role="separator"
      aria-orientation={across ? "vertical" : "horizontal"}
      aria-label={label}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={size || undefined}
      aria-valuetext={
        size ? `${size} pixels ${across ? "wide" : "tall"}` : undefined
      }
      tabIndex={0}
      title={`Drag to resize${onCollapse ? " or past the edge to close" : ""} · double-click to reset`}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        event.preventDefault();
        const start = across ? event.clientX : event.clientY,
          from = current();
        const target = event.currentTarget;
        target.setPointerCapture(event.pointerId);
        setDrag("resize");
        let closing = false;
        const move = (next: PointerEvent) => {
          const moved = (across ? next.clientX : next.clientY) - start;
          const raw = from + sign * moved;
          closing = !!onCollapse && raw < min - COLLAPSE;
          setDrag(closing ? "collapse" : "resize");
          onSize(closing ? from : raw);
        };
        const up = () => {
          target.removeEventListener("pointermove", move);
          target.removeEventListener("pointerup", up);
          target.removeEventListener("pointercancel", up);
          document.body.classList.remove(bodyClass);
          setDrag(null);
          if (closing) onCollapse?.();
        };
        target.addEventListener("pointermove", move);
        target.addEventListener("pointerup", up);
        target.addEventListener("pointercancel", up);
        document.body.classList.add(bodyClass);
      }}
      onDoubleClick={onReset}
      onKeyDown={(event) => {
        const step = event.shiftKey ? 48 : 16;
        if (event.key === less) onSize(current() - step);
        else if (event.key === more) onSize(current() + step);
        else if (event.key === "Home") onSize(min);
        else if (event.key === "End") onSize(max);
        else if (event.key === "Enter") onReset();
        else return;
        event.preventDefault();
      }}
    >
      <span className="edge-resizer-grip" aria-hidden="true" />
      {drag && (
        <span className="edge-resizer-size" aria-hidden="true">
          {drag === "collapse" ? "Release to close" : `${size} px`}
        </span>
      )}
    </div>
  );
}

/**
 * A panel size the person chose, remembered in this browser, or null for the
 * panel's usual size.
 */
export function useStoredSize(
  key: string,
): [number | null, (size: number | null) => void] {
  const [size, setSize] = useState<number | null>(() => {
    try {
      return Number(localStorage.getItem(key)) || null;
    } catch {
      return null;
    }
  });
  const set = (next: number | null) => {
    const value = next === null ? null : Math.round(next);
    setSize(value);
    try {
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, String(value));
    } catch {
      // Without storage the size lasts until the page reloads.
    }
  };
  return [size, set];
}

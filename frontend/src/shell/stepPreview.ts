import { useEffect, useRef } from "react";

/**
 * Tracing a step on the canvas while a row naming it is under the pointer.
 * Rows passed over quickly do not each redraw the canvas, and leaving the
 * tab or closing the panel ends the preview. `rowPreview(id)` gives the
 * pointer handlers for one row.
 */
export function useStepPreview(onPreview?: (id: string | null) => void) {
  useEffect(() => () => onPreview?.(null), [onPreview]);
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  const preview = (id: string | null) => {
    window.clearTimeout(timer.current);
    if (id === null) onPreview?.(null);
    else timer.current = window.setTimeout(() => onPreview?.(id), 140);
  };
  const rowPreview = (id: string | null | undefined) =>
    id && onPreview
      ? {
          onMouseEnter: () => preview(id),
          onMouseLeave: () => preview(null),
        }
      : {};
  return { preview, rowPreview };
}

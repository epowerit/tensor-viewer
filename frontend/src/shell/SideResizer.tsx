import { useState } from "react";
import { EdgeResizer } from "./EdgeResizer";

const KEY = "tensorviewer.sideWidth";
export const SIDE_WIDTH = { min: 200, max: 720, initial: 246 };

/** The widest the side bar may grow: never more than about half the window. */
export function widest(windowWidth: number) {
  return Math.max(
    SIDE_WIDTH.min,
    Math.min(SIDE_WIDTH.max, Math.round(windowWidth * 0.55)),
  );
}

const clamp = (width: number) =>
  Math.round(
    Math.max(
      SIDE_WIDTH.min,
      Math.min(
        typeof window === "undefined" ? SIDE_WIDTH.max : widest(innerWidth),
        width,
      ),
    ),
  );

function stored() {
  try {
    const value = Number(localStorage.getItem(KEY));
    return value ? clamp(value) : SIDE_WIDTH.initial;
  } catch {
    return SIDE_WIDTH.initial;
  }
}

/** The side bar's width, remembered in this browser. */
export function useSideWidth(): [number, (width: number) => void] {
  const [width, setWidth] = useState(stored);
  const set = (next: number) => {
    const value = clamp(next);
    setWidth(value);
    try {
      localStorage.setItem(KEY, String(value));
    } catch {
      // Without storage the width lasts until the page reloads.
    }
  };
  return [width, set];
}

export type SideView = "explorer" | "runs";
const VIEW_KEY = "tensorviewer.sideView";

/**
 * Which view the side bar shows, or null while it is closed, remembered in
 * this browser like an editor's: it reopens as it was left. Where the bar
 * floats over the canvas, on a narrow window, it starts closed. The last view
 * shown is kept too, so toggling the bar brings that view back.
 */
export function useSideView(): [
  SideView | null,
  (view: SideView | null) => void,
  SideView,
] {
  const [state, setState] = useState(() => {
    let saved = "";
    try {
      saved = localStorage.getItem(VIEW_KEY) ?? "";
    } catch {
      // Without storage the bar starts closed on the Projects view.
    }
    const [view, open] = saved.split(":");
    const last: SideView = view === "runs" ? "runs" : "explorer";
    const narrow =
      typeof window !== "undefined" &&
      window.matchMedia?.("(max-width: 760px)").matches;
    return { view: open === "open" && !narrow ? last : null, last };
  });
  const set = (view: SideView | null) => {
    const last = view ?? state.last;
    setState({ view, last });
    try {
      localStorage.setItem(VIEW_KEY, `${last}:${view ? "open" : "closed"}`);
    } catch {
      // The bar keeps its view until the page reloads.
    }
  };
  return [state.view, set, state.last];
}

/** The side bar's edge, in the gap beside it. */
export function SideResizer({
  width,
  onWidth,
  onCollapse,
}: {
  width: number;
  onWidth: (width: number) => void;
  onCollapse?: () => void;
}) {
  return (
    <EdgeResizer
      className="side-resizer"
      label="Resize the side bar"
      grow="right"
      size={width}
      min={SIDE_WIDTH.min}
      max={typeof window === "undefined" ? SIDE_WIDTH.max : widest(innerWidth)}
      onSize={onWidth}
      onReset={() => onWidth(SIDE_WIDTH.initial)}
      onCollapse={onCollapse}
    />
  );
}

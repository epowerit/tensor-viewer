export type CanvasSize = { width: number; height: number };
export type Viewport = { x: number; y: number; scale: number };
export type ViewportFrame = {
  graph: object;
  focusKey: number;
  size: CanvasSize;
};
export type CanvasViewportState = {
  view: Viewport;
  frame: ViewportFrame | null;
  overview: { graph: object; size: CanvasSize; view: Viewport } | null;
};

/** Keep the same tensor coordinates at the center when a drawer resizes the canvas. */
export function resizeViewport(
  view: Viewport,
  previous: CanvasSize,
  next: CanvasSize,
): Viewport {
  if (
    previous.width <= 0 ||
    previous.height <= 0 ||
    next.width <= 0 ||
    next.height <= 0 ||
    (previous.width === next.width && previous.height === next.height)
  )
    return view;
  return {
    ...view,
    x: view.x + (next.width - previous.width) / 2,
    y: view.y + (next.height - previous.height) / 2,
  };
}

/** An explicit fit also becomes the overview used by Back to journey. */
export function fitOverview(
  state: CanvasViewportState,
  graph: object,
  size: CanvasSize,
  view: Viewport,
): CanvasViewportState {
  return { ...state, view, overview: { graph, size, view } };
}

/** Keep overview navigation separate from temporary node inspection. */
export function reframeViewport(
  state: CanvasViewportState,
  next: ViewportFrame,
  fitted: Viewport,
  focus?: (view: Viewport) => Viewport,
): CanvasViewportState {
  if (next.size.width <= 0 || next.size.height <= 0) return state;
  const previous = state.frame;
  let overview = state.overview;
  let view: Viewport;
  if (!previous || previous.graph !== next.graph) {
    // A new layout needs its own overview, even if a node stays selected.
    overview = { graph: next.graph, size: next.size, view: fitted };
    view = next.focusKey > 0 && focus ? focus(state.view) : fitted;
  } else if (previous.focusKey === next.focusKey) {
    view = resizeViewport(state.view, previous.size, next.size);
  } else if (next.focusKey > 0) {
    if (previous.focusKey === 0)
      overview = {
        graph: next.graph,
        size: previous.size,
        view: state.view,
      };
    view = focus ? focus(state.view) : fitted;
  } else {
    view =
      overview?.graph === next.graph
        ? resizeViewport(overview.view, overview.size, next.size)
        : fitted;
  }
  return { view, overview, frame: next };
}

export type Viewport = { x: number; y: number; scale: number };
export type CanvasSize = { width: number; height: number };
export type CanvasFrame = {
  /** The canonical model, stable while a selected operation expands its actors. */
  graph: object;
  /** Display layout may change without replacing the model's saved overview. */
  layout?: object;
  focusKey: number;
  /** Explicit selections frame themselves even with playback following off. */
  selectionKey?: number;
  size: CanvasSize;
};
export type CanvasViewport = {
  view: Viewport;
  frame: CanvasFrame | null;
  overview: { graph: object; size: CanvasSize; view: Viewport } | null;
};

const sameView = (a: Viewport, b: Viewport) =>
  a.x === b.x && a.y === b.y && a.scale === b.scale;
const sameSize = (a: CanvasSize, b: CanvasSize) =>
  a.width === b.width && a.height === b.height;

/** Keep the same world point at the center as adjacent panels open or close. */
export function resizeView(
  view: Viewport,
  before: CanvasSize,
  after: CanvasSize,
): Viewport {
  if (
    before.width <= 0 ||
    before.height <= 0 ||
    after.width <= 0 ||
    after.height <= 0 ||
    (before.width === after.width && before.height === after.height)
  )
    return view;
  return {
    ...view,
    x: view.x + (after.width - before.width) / 2,
    y: view.y + (after.height - before.height) / 2,
  };
}

/** An operation temporarily frames its actors; returning restores the model view. */
export function reframeCanvas(
  current: CanvasViewport,
  frame: CanvasFrame,
  fit: Viewport,
  focus?: (view: Viewport) => Viewport,
  /** Where a canvas opens instead of the overview, such as the last run's camera. */
  start?: Viewport,
): CanvasViewport {
  if (!frame.size.width || !frame.size.height) return current;
  const previous = current.frame;
  let overview = current.overview;
  let view: Viewport;
  if (!previous && start) {
    overview = { graph: frame.graph, size: frame.size, view: fit };
    view = start;
  } else if (!previous || previous.graph !== frame.graph) {
    overview = { graph: frame.graph, size: frame.size, view: fit };
    view = frame.focusKey && focus ? focus(current.view) : fit;
  } else if (
    previous.focusKey === frame.focusKey &&
    previous.selectionKey === frame.selectionKey &&
    (previous.layout ?? previous.graph) === (frame.layout ?? frame.graph)
  ) {
    // An overview nobody has moved stays fitted as the canvas resizes, such
    // as when it was first measured while hidden or before a panel opened.
    const untouched =
      !frame.focusKey &&
      overview?.graph === frame.graph &&
      sameView(current.view, overview.view);
    if (untouched && !sameSize(previous.size, frame.size)) {
      view = fit;
      overview = { graph: frame.graph, size: frame.size, view: fit };
    } else view = resizeView(current.view, previous.size, frame.size);
  } else if (frame.focusKey) {
    if (!previous.focusKey)
      overview = {
        graph: frame.graph,
        size: previous.size,
        view: current.view,
      };
    const resized = resizeView(current.view, previous.size, frame.size);
    view = focus ? focus(resized) : resized;
  } else {
    view =
      overview?.graph === frame.graph
        ? resizeView(overview.view, overview.size, frame.size)
        : fit;
  }
  return { view, frame, overview };
}

/** Below this scale tensor cards show shapes only (see showTensorCells). */
const READABLE = 0.45;

/** Below this scale the overview names only the current and hottest steps. */
const LABELLED = 0.2;

/**
 * The overview of a laid-out journey. A wide canvas fits the whole model. A
 * narrow one (a phone held upright) would shrink a long left-to-right chain
 * until nothing is legible, so it keeps a readable scale instead and starts
 * at the journey's beginning; the rest is a pan away. A wide canvas does the
 * same for a chain so long that fitting it would leave its steps unnamed,
 * unless `whole` asks for everything, as Fit entire journey does.
 */
export function overviewView(
  graph: CanvasSize,
  size: CanvasSize,
  topInset = 0,
  whole = false,
): Viewport {
  const inset = Math.min(topInset, size.height / 3);
  const across = (size.width - 72) / graph.width;
  const down = (size.height - 170 - inset) / graph.height;
  const fitted = Math.min(1, Math.max(0.001, Math.min(across, down)));
  // Centered in the band the scale was fitted to, between the stage's
  // heading and the transport, so a short canvas keeps the model clear of
  // the playback controls.
  const centeredY = (scale: number) =>
    inset + 60 + (size.height - 170 - inset - graph.height * scale) / 2;
  const wide = size.width >= 640;
  if (fitted >= READABLE || (wide && (whole || fitted >= LABELLED)))
    return {
      scale: fitted,
      x: (size.width - graph.width * fitted) / 2,
      y: centeredY(fitted),
    };
  const scale = Math.max(fitted, Math.min(wide ? 0.3 : 0.55, down));
  return { scale, x: wide ? 36 : 12, y: centeredY(scale) };
}

/** Explicit Fit replaces the saved overview as well as the visible framing. */
export function fitCanvas(
  current: CanvasViewport,
  graph: object,
  size: CanvasSize,
  view: Viewport,
): CanvasViewport {
  return { ...current, view, overview: { graph, size, view } };
}

/** Tiny or offscreen cards keep shape metadata; detailed cells appear when readable. */
export function showTensorCells(
  node: { x: number; y: number },
  view: Viewport,
  size: CanvasSize,
  width: number,
  height: number,
): boolean {
  if (view.scale < 0.45) return false;
  const x = node.x * view.scale + view.x;
  const y = node.y * view.scale + view.y;
  const margin = 64;
  return (
    x + width * view.scale >= -margin &&
    y + height * view.scale >= -margin &&
    x <= size.width + margin &&
    y <= size.height + margin
  );
}

// Names used by the overview-restore tests and earlier callers.
export type ViewportFrame = CanvasFrame;
export type CanvasViewportState = CanvasViewport;
export const resizeViewport = resizeView;
export const fitOverview = fitCanvas;
export const reframeViewport = reframeCanvas;

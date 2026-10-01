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

/** Keep the same world point at the center as adjacent panels open or close. */
export function resizeView(
  view: Viewport,
  before: CanvasSize,
  after: CanvasSize,
): Viewport {
  if (!before.width || !before.height || !after.width || !after.height)
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
): CanvasViewport {
  if (!frame.size.width || !frame.size.height) return current;
  const previous = current.frame;
  let overview = current.overview;
  let view: Viewport;
  if (!previous || previous.graph !== frame.graph) {
    overview = { graph: frame.graph, size: frame.size, view: fit };
    view = frame.focusKey && focus ? focus(current.view) : fit;
  } else if (
    previous.focusKey === frame.focusKey &&
    previous.selectionKey === frame.selectionKey &&
    (previous.layout ?? previous.graph) === (frame.layout ?? frame.graph)
  ) {
    view = resizeView(current.view, previous.size, frame.size);
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

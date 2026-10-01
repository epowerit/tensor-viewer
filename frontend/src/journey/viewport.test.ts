import { describe, expect, it } from "vitest";
import {
  fitCanvas,
  reframeCanvas,
  resizeView,
  showTensorCells,
  type CanvasSize,
  type CanvasViewport,
  type CanvasViewportState,
  type Viewport,
  fitOverview,
  reframeViewport,
  resizeViewport,
} from "./viewport";

const graph = {};
const size = { width: 1200, height: 700 };
const fit = { x: 40, y: 60, scale: 0.15 };
const overview = { x: -650, y: -230, scale: 1.2 };
const focused = { x: -200, y: -110, scale: 0.9 };
const state = (): CanvasViewport => ({
  view: overview,
  frame: { graph, size, focusKey: 0 },
  overview: null,
});
const center = (view: typeof overview, frame = size) => [
  (frame.width / 2 - view.x) / view.scale,
  (frame.height / 2 - view.y) / view.scale,
];

describe("journey navigation", () => {
  it("returns to the same overview after inspecting multiple nodes", () => {
    let next = reframeCanvas(
      state(),
      { graph, size, focusKey: 1 },
      fit,
      () => focused,
    );
    next = reframeCanvas(next, { graph, size, focusKey: 2 }, fit, () => ({
      ...focused,
      x: -600,
    }));
    next = reframeCanvas(next, { graph, size, focusKey: 0 }, fit);
    expect(next.view).toEqual(overview);
  });

  it("keeps the same world point when code and shelves resize the overview", () => {
    const narrow = { width: 720, height: 480 };
    const resized = reframeCanvas(
      state(),
      { graph, size: narrow, focusKey: 0 },
      fit,
    );
    expect(center(resized.view, narrow)).toEqual(center(overview));
    expect(resized.view.scale).toBe(overview.scale);
    expect(
      reframeCanvas(resized, { graph, size, focusKey: 0 }, fit).view,
    ).toEqual(overview);
  });

  it("restores the overview center after resizing during inspection", () => {
    const narrow = { width: 720, height: 480 };
    const inspection = reframeCanvas(
      state(),
      { graph, size, focusKey: 1 },
      fit,
      () => focused,
    );
    const resized = reframeCanvas(
      inspection,
      { graph, size: narrow, focusKey: 1 },
      fit,
    );
    expect(center(resized.view, narrow)).toEqual(center(focused));
    const restored = reframeCanvas(
      resized,
      { graph, size: narrow, focusKey: 0 },
      fit,
    );
    expect(center(restored.view, narrow)).toEqual(center(overview));
  });

  it("lets explicit Fit override a stored overview", () => {
    const inspection = reframeCanvas(
      state(),
      { graph, size, focusKey: 1 },
      fit,
      () => focused,
    );
    const fitted = fitCanvas(inspection, graph, size, fit);
    expect(
      reframeCanvas(fitted, { graph, size, focusKey: 0 }, fit).view,
    ).toEqual(fit);
  });

  it("starts a new overview for a new graph or folded stage layout", () => {
    const inspection = reframeCanvas(
      state(),
      { graph, size, focusKey: 1 },
      fit,
      () => focused,
    );
    const nextGraph = {};
    const nextFit = { x: 100, y: 130, scale: 0.7 };
    const changed = reframeCanvas(
      inspection,
      { graph: nextGraph, size, focusKey: 1 },
      nextFit,
      () => focused,
    );
    expect(
      reframeCanvas(changed, { graph: nextGraph, size, focusKey: 0 }, nextFit)
        .view,
    ).toEqual(nextFit);
  });

  it("does not consume framing changes while the canvas is hidden", () => {
    const previous = state();
    const hidden = { width: 0, height: 0 };
    expect(
      reframeCanvas(previous, { graph, size: hidden, focusKey: 1 }, fit),
    ).toBe(previous);
    expect(resizeView(overview, size, hidden)).toBe(overview);
    expect(resizeView(overview, hidden, size)).toBe(overview);
  });
});

it("renders detailed cells only when their node is readable and near the viewport", () => {
  const origin = { x: 0, y: 0 };
  expect(
    showTensorCells(origin, { x: 0, y: 0, scale: 0.1 }, size, 184, 192),
  ).toBe(false);
  expect(
    showTensorCells(origin, { x: 0, y: 0, scale: 0.8 }, size, 184, 192),
  ).toBe(true);
  expect(
    showTensorCells(
      { x: 8000, y: 0 },
      { x: 0, y: 0, scale: 1 },
      size,
      184,
      192,
    ),
  ).toBe(false);
  expect(
    showTensorCells(
      { x: 8000, y: 0 },
      { x: -7800, y: 0, scale: 1 },
      size,
      184,
      192,
    ),
  ).toBe(true);
});

function worldCenter(view: Viewport, size: CanvasSize) {
  return {
    x: (size.width / 2 - view.x) / view.scale,
    y: (size.height / 2 - view.y) / view.scale,
  };
}

describe("canvas resizing", () => {
  it.each([0.001, 0.3, 1, 2])(
    "keeps the inspected world point centered at zoom %s",
    (scale) => {
      const view = { x: -280, y: 123, scale };
      const before = { width: 1280, height: 720 };
      const after = { width: 760, height: 630 };
      const resized = resizeViewport(view, before, after);

      expect(resized.scale).toBe(scale);
      expect(worldCenter(resized, after)).toEqual(worldCenter(view, before));
    },
  );

  it("returns to the same position after opening and closing a drawer", () => {
    const original = { x: -348.25, y: -19, scale: 0.85 };
    const full = { width: 1440, height: 820 };
    const drawer = { width: 1040, height: 820 };
    const mobile = { width: 390, height: 700 };
    let view = resizeViewport(original, full, drawer);
    view = resizeViewport(view, drawer, mobile);
    view = resizeViewport(view, mobile, full);

    expect(view).toEqual(original);
  });

  it("does not move the viewport for an unchanged or hidden frame", () => {
    const view = { x: -40, y: 130, scale: 1.2 };
    const visible = { width: 1200, height: 800 };
    expect(resizeViewport(view, visible, visible)).toBe(view);
    for (const hidden of [
      { width: 0, height: 0 },
      { width: 0, height: 800 },
      { width: 1200, height: 0 },
    ]) {
      expect(resizeViewport(view, visible, hidden)).toBe(view);
      expect(resizeViewport(view, hidden, visible)).toBe(view);
    }
  });
});

describe("returning from tensor inspection", () => {
  const graph = {};
  const size = { width: 1280, height: 720 };
  const fitted = { x: 40, y: 110, scale: 0.3 };
  const overview = { x: -800, y: 130, scale: 1.4 };
  const focused = { x: -190, y: -240, scale: 1.15 };
  const state: CanvasViewportState = {
    view: overview,
    frame: { graph, size, focusKey: 0 },
    overview: null,
  };

  it("restores the user's pan and zoom after visiting multiple nodes", () => {
    let next = reframeViewport(
      state,
      { graph, size, focusKey: 1 },
      fitted,
      () => focused,
    );
    expect(next.view).toEqual(focused);
    next = reframeViewport(
      { ...next, view: { x: -555, y: 90, scale: 2 } },
      { graph, size, focusKey: 2 },
      fitted,
      () => ({ ...focused, x: -990 }),
    );
    next = reframeViewport(next, { graph, size, focusKey: 0 }, fitted);
    expect(next.view).toEqual(overview);
  });

  it("restores the same overview center after a drawer changes canvas size", () => {
    const narrow = { width: 800, height: 650 };
    const next = reframeViewport(
      state,
      { graph, size: narrow, focusKey: 1 },
      fitted,
      () => focused,
    );
    const restored = reframeViewport(
      next,
      { graph, size: narrow, focusKey: 0 },
      fitted,
    );
    expect(worldCenter(restored.view, narrow)).toEqual(
      worldCenter(overview, size),
    );
    expect(restored.view.scale).toBe(overview.scale);
  });

  it("does not replace the saved overview when the focused canvas resizes", () => {
    const narrow = { width: 800, height: 650 };
    let next = reframeViewport(
      state,
      { graph, size, focusKey: 1 },
      fitted,
      () => focused,
    );
    next = reframeViewport(next, { graph, size: narrow, focusKey: 1 }, fitted);
    expect(worldCenter(next.view, narrow)).toEqual(worldCenter(focused, size));
    next = reframeViewport(next, { graph, size, focusKey: 0 }, fitted);
    expect(next.view).toEqual(overview);
  });

  it("lets explicit Fit override the stored overview before focus closes", () => {
    const inspection = reframeViewport(
      state,
      { graph, size, focusKey: 1 },
      fitted,
      () => focused,
    );
    const reset = fitOverview(inspection, graph, size, fitted);
    const next = reframeViewport(reset, { graph, size, focusKey: 0 }, fitted);
    expect(next.view).toEqual(fitted);
  });

  it("starts a fresh overview when graph or stage layout changes", () => {
    const inspection = reframeViewport(
      state,
      { graph, size, focusKey: 1 },
      fitted,
      () => focused,
    );
    const changedGraph = {};
    const newFit = { x: 90, y: 230, scale: 0.75 };
    const next = reframeViewport(
      inspection,
      { graph: changedGraph, size, focusKey: 0 },
      newFit,
    );
    expect(next.view).toEqual(newFit);
    expect(next.overview?.graph).toBe(changedGraph);
  });

  it("uses the new layout's fit when it changes during inspection", () => {
    const inspection = reframeViewport(
      state,
      { graph, size, focusKey: 1 },
      fitted,
      () => focused,
    );
    const changedGraph = {};
    const newFit = { x: 80, y: 50, scale: 0.4 };
    const changed = reframeViewport(
      inspection,
      { graph: changedGraph, size, focusKey: 1 },
      newFit,
      () => ({ ...focused, x: -20 }),
    );
    const restored = reframeViewport(
      changed,
      { graph: changedGraph, size, focusKey: 0 },
      newFit,
    );
    expect(restored.view).toEqual(newFit);
  });

  it("does not consume focus transitions while the canvas is hidden", () => {
    const hidden = reframeViewport(
      state,
      { graph, size: { width: 0, height: 0 }, focusKey: 1 },
      fitted,
      () => focused,
    );
    expect(hidden).toBe(state);
    const shown = reframeViewport(
      hidden,
      { graph, size, focusKey: 1 },
      fitted,
      () => focused,
    );
    expect(shown.overview?.view).toEqual(overview);
  });
});

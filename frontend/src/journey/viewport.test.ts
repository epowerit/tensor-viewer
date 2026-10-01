import { describe, expect, it } from "vitest";
import {
  fitOverview,
  reframeViewport,
  resizeViewport,
  type CanvasSize,
  type CanvasViewportState,
  type Viewport,
} from "./viewport";

function center(view: Viewport, size: CanvasSize) {
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
      expect(center(resized, after)).toEqual(center(view, before));
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
    expect(center(restored.view, narrow)).toEqual(center(overview, size));
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
    expect(center(next.view, narrow)).toEqual(center(focused, size));
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

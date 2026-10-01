import { describe, expect, it } from "vitest";
import {
  fitCanvas,
  reframeCanvas,
  resizeView,
  showTensorCells,
  type CanvasViewport,
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

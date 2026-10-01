import { renderToStaticMarkup } from "react-dom/server";
import type { ComponentProps } from "react";
import { describe, expect, it } from "vitest";
import type { Operation } from "../api/client";
import { SceneTransport } from "./SceneTransport";

const operations = [
  { id: "op0", kind: "linear", outputs: ["y"] },
] as Operation[];
const noop = () => {};
function render(
  overrides: Partial<ComponentProps<typeof SceneTransport>> = {},
) {
  return renderToStaticMarkup(
    <SceneTransport
      operations={operations}
      index={0}
      playing={false}
      busy={false}
      expanded={false}
      speed={1}
      reveal={false}
      following
      onPlay={noop}
      onSeek={noop}
      onInspect={noop}
      onOverview={noop}
      onSpeed={noop}
      onReveal={noop}
      onFollow={noop}
      {...overrides}
    >
      {null}
    </SceneTransport>,
  );
}
function button(markup: string, label: string) {
  return markup.match(
    new RegExp(`<button[^>]*aria-label="${label}"[^>]*>`),
  )?.[0];
}

describe("canvas transport navigation", () => {
  it("lets input inspection return to the canvas before any operation", () => {
    const html = render({ index: -1, expanded: true, frameLabel: "Input x" });
    expect(button(html, "Return to model canvas")).toBeDefined();
    expect(button(html, "Return to model canvas")).not.toContain("disabled");
    expect(button(html, "Previous operation")).toContain("disabled");
    expect(button(html, "Next operation")).not.toContain("disabled");
  });
  it("names a selected stage and allows entering its first operation", () => {
    const html = render({
      frameLabel: "Feature extractor",
      framePosition: "1–1 / 1",
      nextIndex: 0,
      expanded: true,
    });
    expect(html).toContain("Feature extractor");
    expect(html).toContain("1–1 / 1");
    expect(button(html, "Next operation")).not.toContain("disabled");
  });
  it("can inspect an input even when the recording has no operations", () => {
    const html = render({
      operations: [],
      index: -1,
      canInspect: true,
      inspectLabel: "Inspect current tensor",
      frameLabel: "Input x",
    });
    expect(button(html, "Inspect current tensor")).not.toContain("disabled");
    expect(button(html, "Inspect current tensor")).toContain(
      'aria-expanded="false"',
    );
    expect(button(html, "Play tensor journey")).toContain("disabled");
  });
  it("disables execution navigation for a recording with no operations", () => {
    const html = render({ operations: [], index: -1 });
    for (const name of [
      "Previous operation",
      "Next operation",
      "Play tensor journey",
      "Inspect current operation",
    ])
      expect(button(html, name)).toContain("disabled");
  });
  it("keeps tracking notices discoverable before opening options", () => {
    expect(button(render({ notices: 2 }), "Playback options")).toContain(
      "2 tracking notices",
    );
    expect(button(render({ stopped: true }), "Playback options")).toContain(
      "run stopped",
    );
  });
});

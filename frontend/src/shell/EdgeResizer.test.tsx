import { renderToStaticMarkup } from "react-dom/server";
import { expect, test } from "vitest";
import { EdgeResizer } from "./EdgeResizer";

const edge = (grow: "right" | "left" | "up", size: number) =>
  renderToStaticMarkup(
    <EdgeResizer
      className="test-resizer"
      label="Resize"
      grow={grow}
      size={size}
      min={120}
      max={600}
      onSize={() => {}}
      onReset={() => {}}
      onCollapse={() => {}}
    />,
  );

test("a side edge is a vertical separator that reads its width", () => {
  const html = edge("right", 246);
  expect(html).toContain('aria-orientation="vertical"');
  expect(html).toContain('aria-valuetext="246 pixels wide"');
  expect(html).toContain("or past the edge to close");
});

test("a bottom edge is a horizontal separator that reads its height", () => {
  const html = edge("up", 240);
  expect(html).toContain('aria-orientation="horizontal"');
  expect(html).toContain('aria-valuetext="240 pixels tall"');
});

test("an edge without a size of its own yet claims none", () => {
  const html = edge("left", 0);
  expect(html).not.toContain("aria-valuenow");
  expect(html).toContain("edge-resizer-grip");
});

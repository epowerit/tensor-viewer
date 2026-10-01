import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { CanvasCellProbe, type CanvasProbe } from "./CanvasCellProbe";
import type { Tensor } from "../api/client";

const tensor = {
  id: "result",
  name: "result",
  shape: [2],
  numel: 2,
  value_source: "inline",
  values: [3.25, 7.5],
} as Tensor;
function render(source: Tensor, partial = false) {
  const probe: CanvasProbe = {
    tensor: source,
    index: 1,
    runId: "recording",
    result: {
      status: "mapped",
      sources: [{ tensorId: "input", index: 2, role: "input" }],
      total: partial ? 10000 : 1,
      truncated: partial,
      summary: "Sampled source cells are highlighted.",
    },
  };
  return renderToStaticMarkup(
    <CanvasCellProbe probe={probe} onInspect={() => {}} onClear={() => {}} />,
  );
}

it("reads the selected snapshot coordinate rather than the first tensor value", () => {
  const html = render(tensor);
  expect(html).toContain("= 7.5");
  expect(html).not.toContain("3.25");
});
it("does not present unavailable or shape-only values as recorded numbers", () => {
  const shape = render({ ...tensor, value_source: "shape" });
  expect(shape).toContain("shape only");
  expect(shape).not.toContain("7.5");
  expect(render({ ...tensor, values: [] })).toContain("value unavailable");
  expect(
    render({ ...tensor, value_source: "paged", values: [] }),
  ).not.toContain("= 0");
});
it("keeps the bounded coverage count ahead of the explanation", () => {
  const html = render(tensor, true);
  expect(html).toContain("1 of 10,000 source uses tracked");
  expect(html.indexOf("source uses tracked")).toBeLessThan(
    html.indexOf("Sampled source cells", html.indexOf("<b>")),
  );
});

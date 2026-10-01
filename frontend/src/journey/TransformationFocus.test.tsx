import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { Run, Tensor } from "../api/client";
import type { JourneyNode } from "./graph";
import { TransformationFocus } from "./TransformationFocus";
import { JourneyInspector } from "./JourneyInspector";

const tensor = {
  id: "input",
  name: "x",
  role: "input",
  shape: [2, 2, 12],
  axes: ["batch", "tokens", "features"],
  strides: [24, 12, 1],
  storage_offset: 0,
  numel: 48,
  dtype: "float32",
  contiguous: true,
  value_source: "inline",
  values: Array.from({ length: 48 }, (_, index) => index),
} as Tensor;
const node = { id: "input-input", tensors: [tensor] } as JourneyNode;
const run = {
  id: "recording",
  project: { code: "", input: { generator: "arange" } },
  trace: { operations: [], tensors: { input: tensor } },
} as unknown as Run;
const noop = () => {};

function render(
  view: "focus" | "inspector",
  tensorId = "input",
  source = tensor,
) {
  const props = {
    run,
    node: { ...node, tensors: [source] },
    initialTensorId: tensorId,
    initialCell: 47,
    showValues: true,
    onShowValues: noop,
    onSelect: noop,
    onClose: noop,
  };
  return renderToStaticMarkup(
    view === "focus" ? (
      <TransformationFocus
        {...props}
        active
        connections={null}
        codeOpen={false}
        inspectorOpen={false}
        onCode={noop}
      />
    ) : (
      <JourneyInspector {...props} tab="values" onTab={noop} />
    ),
  );
}

for (const view of ["focus", "inspector"] as const) {
  describe(`${view} input cell inspection`, () => {
    it("opens the selected coordinate's slice and window", () => {
      const html = render(view);
      expect(html).toMatch(/data-cell-index="47"[^>]*aria-pressed="true"/);
      expect(html).toContain("[1, 1, 11]");
      expect(html).toContain("Cols 8–11");
    });

    it("does not apply a cell that belongs to a different tensor", () => {
      const html = render(view, "different-tensor");
      expect(html).toMatch(/data-cell-index="0"[^>]*aria-pressed="true"/);
      expect(html).toContain("[0, 0, 0]");
    });

    it("describes shape-only snapshots without offering numeric values", () => {
      const html = render(view, "input", {
        ...tensor,
        value_source: "shape",
        values: [],
      });
      expect(html).toContain("Shapes only · no numeric values");
      expect(html).not.toContain('type="checkbox"');
    });
  });
}

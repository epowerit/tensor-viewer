import { renderToStaticMarkup } from "react-dom/server";
import { expect, test } from "vitest";
import type { Run, Tensor } from "../api/client";
import { CellExperiments } from "./CellExperiments";
import { TensorUseContext, tensorUses } from "./TensorUseContext";
import { WhatIfContext, type WhatIfControl } from "./WhatIf";

const tensor = (id: string, name: string, extra = {}) =>
  ({
    id,
    name,
    shape: [2, 3],
    numel: 6,
    dtype: "float32",
    axes: ["batch", "features"],
    storage_id: `s-${id}`,
    role: "intermediate",
    value_source: "inline",
    values: [0, 1, 2, 3, 4, 5],
    ...extra,
  }) as unknown as Tensor;

const trace = {
  input_ids: ["x"],
  output_ids: ["y"],
  operations: [
    { id: "op0", index: 0, kind: "relu", inputs: ["x"], outputs: ["y"] },
  ],
  tensors: {
    x: tensor("x", "x", { role: "input" }),
    y: tensor("y", "y"),
    w: tensor("w", "weight", { role: "parameter" }),
  },
} as unknown as Run["trace"];

const control = {
  inputId: "x",
  vocabulary: null,
  edits: [],
  recorded: () => undefined,
  busy: false,
  run: () => {},
  learn: () => {},
  learned: null,
  knockout: () => {},
  knocked: null,
  baseRunId: "run",
  patchFrom: null,
  sweep: async () => null,
  leave: null,
} satisfies WhatIfControl;

const render = (id: string, withControl = true) =>
  renderToStaticMarkup(
    <TensorUseContext
      value={{ uses: tensorUses(trace), go: () => {}, trace, runId: "run" }}
    >
      <WhatIfContext value={withControl ? control : null}>
        <CellExperiments tensor={trace.tensors[id]} index={0} coords={[0, 0]} />
      </WhatIfContext>
    </TensorUseContext>,
  );

test("a step's result offers every experiment, closed until chosen", () => {
  const html = render("y");
  expect(html).toContain("What moves it");
  expect(html).toContain("Train on it");
  expect(html).toContain("Knock out");
  expect(html).not.toContain('aria-selected="true"');
});

test("an input or a weight offers nothing to ask", () => {
  // The input is what the others are measured against, and a weight is set
  // before the input arrives and made by no step.
  expect(render("x")).toBe("");
  expect(render("w")).toBe("");
});

test("without a what-if run to vary, only the gradient question stays", () => {
  const html = render("y", false);
  expect(html).toContain("What moves it");
  expect(html).not.toContain("Train on it");
  expect(html).not.toContain("Knock out");
});

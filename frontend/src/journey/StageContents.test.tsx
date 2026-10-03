import { renderToStaticMarkup } from "react-dom/server";
import { expect, test } from "vitest";
import type { Operation, Tensor } from "../api/client";
import { StageContents } from "./StageContents";
import type { JourneyStage } from "./stages";

const tensor = (id: string, name: string, shape: number[]) =>
  ({ id, name, shape }) as Tensor;
const op = (id: string, kind: string, output: string, line: number) =>
  ({
    id,
    kind,
    outputs: [output],
    source: { line },
    status: "ok",
  }) as Operation;

test("a folded card lists its signature and each step's result", () => {
  const tensors = Object.fromEntries(
    [
      tensor("x", "x", [1, 13, 16]),
      tensor("a", "linear", [1, 13, 48]),
      tensor("q", "q", [1, 13, 16]),
    ].map((t) => [t.id, t]),
  );
  const html = renderToStaticMarkup(
    <StageContents
      stage={
        {
          title: "Attention",
          operationIds: ["op0", "op1"],
          inputs: ["x"],
          outputs: ["q"],
        } as JourneyStage
      }
      operations={[op("op0", "linear", "a", 27), op("op1", "chunk", "q", 27)]}
      tensors={tensors}
    />,
  );
  expect(html).toContain("<b>Attention</b>");
  expect(html).toContain("2 steps");
  expect(html).toContain("x [1, 13, 16] → [1, 13, 16]");
  expect(html).toContain("<code>[1, 13, 48]</code>");
  expect(html).toContain("<b> q</b>");
});

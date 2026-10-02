import { renderToStaticMarkup } from "react-dom/server";
import { expect, test } from "vitest";
import type { Run } from "../api/client";
import {
  describeAxis,
  lineageOf,
  originSummary,
  restatesAxis,
  shortAxis,
} from "./axisLineage";
import recorded from "./fixtures/lineage-trace.json";
import { LineageContext } from "./LineageContext";
import { TensorCard } from "./TensorCard";

const trace = recorded as unknown as Run["trace"];
const byName = (name: string) =>
  Object.values(trace.tensors).find((tensor) => tensor.name === name)!;
const render = (name: string, provided = true) =>
  renderToStaticMarkup(
    <LineageContext value={provided ? lineageOf(trace) : null}>
      <TensorCard
        tensor={byName(name)}
        label="Result"
        showValues={false}
        gridFrame={{ rows: 4, columns: 8 }}
      />
    </LineageContext>,
  );

test("short origins fit an axis badge", () => {
  const heads = lineageOf(trace)(byName("heads").id);
  expect(heads.map(shortAxis)).toEqual([
    "x.batch",
    "x.features[1/2]",
    "x.tokens",
    "x.features[2/2]",
  ]);
  expect(heads.map(describeAxis)[1]).toBe("x.features (piece 1 of 2×4)");
});

test("badges drop repeated names and long weight paths", () => {
  const story = (label: string, role: "input" | "parameter") => ({
    size: 4,
    note: null,
    terms: [{ label, size: 4, role }],
  });
  expect(shortAxis(story("tokens.tokens", "input"))).toBe("tokens");
  expect(
    shortAxis(story("text_encoder.block.qkv.weight axis 0", "parameter")),
  ).toBe("qkv.weight axis 0");
  expect(shortAxis(story("encoder.weight.features", "parameter"))).toBe(
    "encoder.weight.features",
  );
  expect(shortAxis(story("x axis 2", "input"))).toBe("x axis 2");
  // A bare index keeps the name it belongs to.
  expect(shortAxis(story("moe.experts.0.2.weight axis 0", "parameter"))).toBe(
    "experts.0.2.weight axis 0",
  );
  expect(shortAxis(story("blocks.1.mlp.weight axis 1", "parameter"))).toBe(
    "mlp.weight axis 1",
  );
});

test("axis badges say where a made axis came from", () => {
  const html = render("heads");
  expect(html).toContain("← x.features[1/2]");
  expect(html).toContain("← x.tokens");
  // Without a run's lineage, badges keep only the tensor's own axes.
  expect(render("heads", false)).not.toContain("axis-origin");
});

test("a tensor that is its own source gets no origins", () => {
  expect(render("x")).not.toContain("axis-origin");
});

test("a tensor's origin says repeated origins once", () => {
  const made = { size: 4, note: "computed by conv2d", terms: [] };
  const batch = {
    size: 2,
    note: null,
    terms: [{ label: "x.batch", size: 2, role: "input" as const }],
  };
  expect(originSummary([batch, made, made, made])).toBe(
    "x.batch · computed by conv2d",
  );
});

test("an origin that repeats the axis's own name is not a badge", () => {
  const from = (label: string, part?: boolean) => ({
    size: 4,
    note: null,
    terms: [
      {
        label,
        size: 4,
        role: "input" as const,
        ...(part ? { part: { index: 0, of: 2, sizes: [2, 2] } } : {}),
      },
    ],
  });
  expect(restatesAxis(from("x.batch"), "batch")).toBe(true);
  expect(restatesAxis(from("x.tokens"), "heads")).toBe(false);
  expect(restatesAxis(from("x.features", true), "features")).toBe(false);
  expect(restatesAxis(from("x.tokens"), "axis 1")).toBe(false);
});

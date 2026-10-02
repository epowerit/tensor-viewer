import { renderToStaticMarkup } from "react-dom/server";
import { expect, test } from "vitest";
import type { Run } from "../api/client";
import recorded from "../tensors/fixtures/lineage-trace.json";
import { VariablesPanel } from "./VariablesPanel";

const run = { trace: recorded } as unknown as Run;
const html = renderToStaticMarkup(
  <VariablesPanel run={run} selected={null} onSelect={() => {}} />,
);
const card = (name: string) =>
  html
    .split("<li>")
    .find((item) => item.includes(`aria-label="Inspect ${name},`)) ?? "";

test("a card says where a reshaped tensor's axes come from", () => {
  expect(card("heads")).toContain(
    "axes from x.batch · x.features (piece 1 of 2×4) · x.tokens · x.features (piece 2 of 2×4)",
  );
  expect(card("heads")).toContain("tensor-shelf-lineage");
  // The visible line is badge-sized; the full story stays in the label.
  expect(card("heads")).toContain(
    "← x.batch · x.features[1/2] · x.tokens · x.features[2/2]",
  );
});

test("an input is its own source and gets no lineage line", () => {
  expect(card("x")).not.toBe("");
  expect(card("x")).not.toContain("tensor-shelf-lineage");
});

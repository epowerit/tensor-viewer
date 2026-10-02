import { expect, test } from "vitest";
import type { Run } from "../api/client";
import { INK_PALETTE, NEUTRAL_INK, inkOf, inkOfAll } from "./axisInk";
import recorded from "./fixtures/lineage-trace.json";

const trace = recorded as unknown as Run["trace"];
const byName = (name: string) =>
  Object.values(trace.tensors).find((tensor) => tensor.name === name)!;
const [batch, tokens, features] = INK_PALETTE;
const ink = inkOf(trace);
const colors = (name: string) => ink(byName(name))!.map((item) => item?.colors);

test("each input axis has its own ink, and it travels with the axis", () => {
  expect(colors("x")).toEqual([[batch], [tokens], [features]]);
  // Split into heads and moved: both pieces keep the features ink.
  expect(colors("heads")).toEqual([[batch], [features], [tokens], [features]]);
  expect(ink(byName("heads"))!.map((item) => item?.piece)).toEqual([
    null,
    "1/2",
    null,
    "2/2",
  ]);
  // Put back together, it is simply features again.
  expect(colors("merged")).toEqual([[batch], [tokens], [features]]);
});

test("weights are neutral and untraced axes have no ink", () => {
  // proj = x @ w: the last axis comes from the weight.
  expect(colors("proj")).toEqual([[batch], [tokens], [NEUTRAL_INK]]);
  // cat joins parts: the joined axis has no single story.
  expect(colors("both")[2]).toBeUndefined();
});

test("a tensor is inked only by the trace it belongs to", () => {
  // Lessons relabel axes on a copy: still the same tensor.
  const relabeled = { ...byName("heads"), axes: ["b", "h", "t", "d"] };
  expect(ink(relabeled)).toEqual(ink(byName("heads")));
  // Another trace's tensor that reuses the id is not.
  const stranger = { ...byName("heads"), storage_id: "elsewhere" };
  expect(ink(stranger)).toBeNull();
  expect(inkOfAll([null, trace])(byName("heads"))).not.toBeNull();
});

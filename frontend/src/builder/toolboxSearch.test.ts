import { describe, expect, it } from "vitest";
import type { ToolboxItem } from "../api/client";
import { filterToolboxItems, matchesInputTool } from "./toolboxSearch";

function tool(
  kind: string,
  title: string,
  description: string,
  group = "Layers",
): ToolboxItem {
  return { kind, title, description, group, parameters: [] };
}

// Names and descriptions mirror the built-in catalog; only irrelevant settings
// are omitted, so searches exercise the words users actually encounter.
const catalog: ToolboxItem[] = [
  tool(
    "attention",
    "Self-attention",
    "Explicit Q, K, V projections and attention heads.",
    "Models",
  ),
  tool(
    "conv2d",
    "2D convolution",
    "A spatial convolution over [batch, channels, height, width].",
    "Spatial",
  ),
  tool(
    "rnn",
    "Simple RNN",
    "An explicit tanh recurrent layer. Up to 32 time steps in this release.",
    "Sequence",
  ),
  tool(
    "linear",
    "Linear projection",
    "Project the final dimension into a new feature space.",
  ),
  tool(
    "conv1d",
    "1D convolution",
    "Convolution over [batch, channels, length]. Padding is kernel size // 2.",
    "Sequence",
  ),
  tool(
    "split_axis",
    "Split axis",
    "Reshape one dimension into two factors, for example features → heads × head features.",
    "Shape adapters",
  ),
  tool(
    "merge_axes",
    "Merge axes",
    "Reshape two adjacent dimensions into one.",
    "Shape adapters",
  ),
  tool(
    "transpose",
    "Transpose axes",
    "Swap two axes without changing their values.",
    "Shape adapters",
  ),
  tool(
    "flatten",
    "Flatten features",
    "Keep the batch axis; combine all remaining axes.",
    "Shape adapters",
  ),
  tool(
    "gelu",
    "GELU",
    "Smooth Gaussian error linear activation.",
    "Activations",
  ),
  {
    ...tool("custom", "Custom display", "Saved copy", "Custom"),
    custom: {
      id: "feature-scale",
      created_at: "today",
      name: "Feature scaling",
      description: "Multiply features by a configurable scale.",
      class_name: "FeatureScale",
      constructor: { scale: 2 },
      code: "class FeatureScale: pass # NotSearchableSourceComment",
    },
  },
];
const kinds = (query: string) =>
  filterToolboxItems(catalog, query).map((item) => item.kind);

describe("toolbox search", () => {
  it("treats blank whitespace and punctuation as an empty search", () => {
    for (const query of ["", "  \n\t ", "—,._"])
      expect(filterToolboxItems(catalog, query)).toBe(catalog);
  });

  it("normalizes case, punctuation, and spacing while requiring every word", () => {
    expect(kinds("  SELF---attention  ")).toEqual(["attention"]);
    expect(kinds("projection, LINEAR")).toEqual(["linear"]);
    expect(kinds("linear\nprojection\tunknown")).toEqual([]);
    expect(kinds("projection lin")).toEqual(["linear"]);
    expect(kinds("ｄｅｎｓｅ")).toEqual(["linear"]);
  });

  it("offers actual convolution tools for CNN with dimensional filtering", () => {
    expect(kinds("CNN")).toEqual(["conv2d", "conv1d"]);
    expect(kinds("CNN 1D")).toEqual(["conv1d"]);
    expect(kinds("2d convolutional")).toEqual(["conv2d"]);
    expect(kinds("conv_2d")).toEqual(["conv2d"]);
    expect(kinds("CNN 3d")).toEqual([]);
  });

  it("supports familiar names for the existing linear, recurrent, and attention tools", () => {
    for (const query of ["linear layer", "dense", "fully-connected"])
      expect(kinds(query)).toEqual(["linear"]);
    for (const query of ["RNN", "recurrent", "neural recurrent"])
      expect(kinds(query)).toEqual(["rnn"]);
    for (const query of [
      "multi head",
      "head multi attention",
      "multi-head self-attention",
      "multihead",
      "MHA",
    ])
      expect(kinds(query)).toEqual(["attention"]);
    expect(kinds("LSTM")).toEqual([]);
    expect(kinds("GRU")).toEqual([]);
  });

  it("finds the constrained reshape adapters without implying arbitrary reshapes", () => {
    expect(kinds("reshape")).toEqual(["split_axis", "merge_axes"]);
    expect(kinds("reshape factors")).toEqual(["split_axis"]);
    expect(kinds("adjacent reshape")).toEqual(["merge_axes"]);
    expect(kinds("arbitrary reshape")).toEqual([]);
    expect(kinds("permute")).toEqual([]);
  });

  it("searches custom titles, descriptions, and class names without scanning source code", () => {
    for (const query of [
      "feature scaling",
      "scale configurable",
      "FeatureScale",
      "featurescale",
      "scale feature",
      "display custom",
    ])
      expect(kinds(query)).toEqual(["custom"]);
    expect(kinds("NotSearchableSourceComment")).toEqual([]);
  });

  it("preserves input catalog order and does not mutate item metadata", () => {
    const reversed = [...catalog].reverse();
    const before = JSON.stringify(reversed);
    const results = filterToolboxItems(reversed, "CNN");
    expect(results.map((item) => item.kind)).toEqual(["conv1d", "conv2d"]);
    expect(results[0]).toBe(reversed.find((item) => item.kind === "conv1d"));
    expect(JSON.stringify(reversed)).toBe(before);
  });
});

describe("input tool search", () => {
  it("matches multiword shape and value queries in any order", () => {
    for (const query of [
      "input tensor",
      "VALUES — STARTING",
      "dimensions input",
      "shape",
      "ten inp",
    ])
      expect(matchesInputTool(query)).toBe(true);
  });

  it("leaves blank-query visibility to the category UI and rejects unrelated tools", () => {
    for (const query of [
      "",
      "  ",
      "...",
      "CNN",
      "input attention",
      "tensor invalid",
    ])
      expect(matchesInputTool(query)).toBe(false);
  });
});

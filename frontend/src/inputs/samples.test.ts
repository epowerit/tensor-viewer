import { expect, test } from "vitest";
import type { Draft, Tensor } from "../api/client";
import { inputIssue } from "./fixtures";
import {
  pixelColors,
  pixelPlan,
  tokenize,
  withGenerator,
  withSentence,
} from "./samples";

const input: Draft["input"] = {
  shape: [2, 4, 8],
  axis_names: ["batch", "tokens", "features"],
  generator: "arange",
  dtype: "float32",
  seed: 7,
};

test("sentences tokenize exactly as the backend does", () => {
  // The same fixture is asserted in backend/tests/test_samples.py.
  const result = tokenize("The cat sat on the mat.");
  expect(result.tokens).toEqual(["the", "cat", "sat", "on", "the", "mat", "."]);
  expect(result.vocabulary).toEqual([".", "cat", "mat", "on", "sat", "the"]);
  expect(result.ids).toEqual([5, 1, 4, 3, 5, 2, 0]);
  expect(tokenize("don't  stop!").tokens).toEqual(["don't", "stop", "!"]);
  expect(tokenize("   ").tokens).toEqual([]);
});

test("Unicode symbols keep the backend's token boundaries, order, and ids", () => {
  expect(tokenize("a 😀 a")).toEqual({
    tokens: ["a", "😀", "a"],
    vocabulary: ["a", "😀"],
    ids: [0, 1, 0],
  });
  // UTF-16 would incorrectly sort the emoji before this BMP character.
  expect(tokenize("😀 \ue000 😀")).toEqual({
    tokens: ["😀", "\ue000", "😀"],
    vocabulary: ["\ue000", "😀"],
    ids: [1, 0, 1],
  });
  expect(withSentence(input, "a 😀 a").shape).toEqual([1, 3]);
  expect(inputIssue(withSentence(input, "😀".repeat(64)))).toBe("");
  expect(inputIssue(withSentence(input, "😀".repeat(65)))).toContain("64");
});

test("sentence whitespace matches Python, including NEL and retained BOM", () => {
  expect(tokenize("a\u0085b\u001cc\ufeffd")).toEqual({
    tokens: ["a", "b", "c", "\ufeff", "d"],
    vocabulary: ["a", "b", "c", "d", "\ufeff"],
    ids: [0, 1, 2, 4, 3],
  });
});

test("changing the value source keeps the input valid", () => {
  const text = withGenerator(input, "text");
  expect(text).toMatchObject({
    shape: [1, 6],
    dtype: "int64",
    axis_names: ["batch", "tokens"],
  });
  expect(inputIssue(text)).toBe("");
  expect(withSentence(text, "hello big world !").shape).toEqual([1, 4]);
  const image = withGenerator(text, "image");
  expect(image).toMatchObject({
    shape: [1, 3, 16, 16],
    dtype: "float32",
    text: null,
    axis_names: ["batch", "channels", "height", "width"],
  });
  expect(inputIssue(image)).toBe("");
  // A user's own image-like shape and names are kept.
  const custom = withGenerator(
    { ...input, shape: [4, 1, 12, 10], axis_names: [] },
    "image",
  );
  expect(custom.shape).toEqual([4, 1, 12, 10]);
  expect(custom.axis_names).toEqual(["batch", "channels", "height", "width"]);
  expect(withGenerator(text, "random").dtype).toBe("float32");
  expect(withGenerator(text, "arange").dtype).toBe("int64");
  expect(inputIssue({ ...text, shape: [1, 5] })).toContain("6 tokens");
  expect(
    inputIssue({ ...input, generator: "image", shape: [8], axis_names: [] }),
  ).toContain("height and width");
  expect(inputIssue({ ...input, text: "stray" })).toContain("need a sentence");
});

const picture = (shape: number[], axes: string[], dtype = "float32") =>
  ({
    shape,
    axes,
    dtype,
    numel: shape.reduce((a, b) => a * b, 1),
  }) as Tensor;

test("pictures are drawn only from explicitly named height and width", () => {
  const names = ["batch", "channels", "height", "width"];
  const rgb = pixelPlan(picture([2, 3, 2, 2], names), 12)!;
  expect(rgb).toMatchObject({ color: true, prefix: [1], height: 2, width: 2 });
  expect(rgb.pixels[0]).toEqual([12, 16, 20]);
  expect(rgb.pixels[3]).toEqual([15, 19, 23]);
  // Eight feature maps are not colors: show the selected one in gray.
  const features = pixelPlan(picture([1, 8, 2, 2], names), 21)!;
  expect(features).toMatchObject({ color: false, prefix: [0, 5] });
  expect(features.pixels.flat()).toEqual([20, 21, 22, 23]);
  expect(pixelPlan(picture([2, 3, 2, 2], []), 0)).toBeNull();
  expect(
    pixelPlan(picture([3, 8, 8], ["axis 0", "axis 1", "axis 2"]), 0),
  ).toBeNull();
  expect(
    pixelPlan(picture([3, 64, 64], ["channels", "height", "width"]), 0),
  ).toBeNull();
  expect(pixelPlan(picture([4, 4], ["height", "width"], "bool"), 0)).toBeNull();
  expect(pixelPlan(picture([4, 4], ["height", "width"]), 0)!.prefix).toEqual(
    [],
  );
});

test("pixel colors use the plane's own range and skip missing values", () => {
  const plan = pixelPlan(picture([2, 2], ["height", "width"]), 0)!;
  const values = [0, 5, 10, "nan"];
  const result = pixelColors(plan, (i) => values[i])!;
  expect(result.colors).toEqual([
    "rgb(0, 0, 0)",
    "rgb(128, 128, 128)",
    "rgb(255, 255, 255)",
    null,
  ]);
  expect([result.minimum, result.maximum]).toEqual([0, 10]);
  expect(pixelColors(plan, () => undefined)).toBeNull();
  const flat = pixelColors(plan, () => 3)!;
  expect(flat.colors[0]).toBe("rgb(0, 0, 0)");
});

test("finite extremes and subnormals produce valid pixel colors", () => {
  const plan = pixelPlan(picture([2, 2], ["height", "width"]), 0)!;
  for (const magnitude of [Number.MAX_VALUE, 1e308, Number.MIN_VALUE]) {
    const values = [-magnitude, 0, magnitude, "nan"];
    expect(pixelColors(plan, (i) => values[i])?.colors).toEqual([
      "rgb(0, 0, 0)",
      "rgb(128, 128, 128)",
      "rgb(255, 255, 255)",
      null,
    ]);
  }
});

import { expect, test } from "vitest";
import type { Draft } from "../api/client";
import { forwardInputs, withInputAt } from "./InputBar";

const input = (shape: number[]) =>
  ({
    shape,
    generator: "random",
    dtype: "float32",
    axis_names: [],
  }) as unknown as Draft["input"];

test("each forward input is edited in its own slot", () => {
  const draft = {
    input_name: "image",
    input: input([2, 3]),
    additional_inputs: [
      { name: "tokens", binding: "positional", input: input([2, 7]) },
      { name: "mask", binding: "keyword", input: input([2, 7]) },
    ],
  } as unknown as Draft;
  expect(forwardInputs(draft).map((item) => item.name)).toEqual([
    "image",
    "tokens",
    "mask",
  ]);
  const tokens = withInputAt(draft, 1, input([2, 9]));
  expect(tokens.additional_inputs?.[0].input.shape).toEqual([2, 9]);
  expect(tokens.additional_inputs?.[1]).toBe(draft.additional_inputs?.[1]);
  expect(tokens.input).toBe(draft.input);
  expect(withInputAt(draft, 0, input([4])).input.shape).toEqual([4]);
});

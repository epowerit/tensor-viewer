import type { Draft, InputFixtureDraft } from "../api/client";
import { MAX_TOKENS, tokenize } from "./samples";

export function inputIssue(input: Draft["input"], mode = "values"): string {
  if ((input.generator === "uploaded") !== !!input.uploaded)
    return "Choose a saved tensor file for uploaded values.";
  if (
    input.uploaded &&
    (input.dtype !== input.uploaded.dtype ||
      input.shape.join(",") !== input.uploaded.shape.join(","))
  )
    return "Uploaded shapes and data types come from the file. Transform them in your model.";
  const count = input.shape.reduce((a, b) => a * b, 1);
  if (
    !input.shape.length ||
    input.shape.length > 6 ||
    input.shape.some((n) => !Number.isInteger(n) || n < 1) ||
    count > 2 ** 40
  )
    return "Use 1–6 positive dimensions, up to 2^40 logical elements.";
  if (mode === "values" && count > 8_388_608)
    return "Use Shapes only for more than 8,388,608 input elements.";
  if (input.axis_names.length && input.axis_names.length !== input.shape.length)
    return "Supply one name per axis, or leave axis names blank.";
  if (
    !Number.isInteger(input.seed) ||
    input.seed < 0 ||
    input.seed > 4294967295
  )
    return "Use an integer seed from 0 to 4,294,967,295.";
  if (input.generator === "random" && input.dtype === "int64")
    return "Random normal inputs need a floating-point data type.";
  if (
    input.generator === "image" &&
    (input.shape.length < 2 || input.dtype === "int64")
  )
    return "A sample image needs height and width as its last two axes and a floating-point data type.";
  if ((input.generator === "text") !== (input.text != null))
    return "Sentence inputs need a sentence.";
  if (input.text != null) {
    const count = tokenize(input.text).tokens.length;
    if (count < 1 || count > MAX_TOKENS)
      return `Use a sentence with 1 to ${MAX_TOKENS} words and symbols.`;
    if (input.shape.join() !== `1,${count}` || input.dtype !== "int64")
      return `This sentence has ${count} tokens: its shape is [1, ${count}] and its data type is int64.`;
  }
  return "";
}

/** Copy settings: the library never holds a live reference to a project draft. */
export function fixtureSettings(
  fixture: Pick<InputFixtureDraft, "input" | "capture_mode">,
) {
  return {
    input: {
      ...fixture.input,
      shape: [...fixture.input.shape],
      axis_names: [...fixture.input.axis_names],
      ...(fixture.input.uploaded
        ? {
            uploaded: {
              ...fixture.input.uploaded,
              shape: [...fixture.input.uploaded.shape],
            },
          }
        : {}),
    },
    capture_mode: fixture.capture_mode,
  };
}

export const generatorLabels: Record<Draft["input"]["generator"], string> = {
  arange: "Sequential",
  random: "Random normal",
  ones: "Ones",
  zeros: "Zeros",
  uploaded: "Uploaded .npy",
  image: "Sample image",
  text: "Sentence tokens",
};

export function uploadIssue(file: { name: string; size: number }): string {
  if (!file.name.toLowerCase().endsWith(".npy"))
    return "Choose a NumPy .npy tensor file.";
  if (file.name.length > 200)
    return "Use a file name with at most 200 characters.";
  if (!file.size) return "This file is empty.";
  if (file.size > 8_388_608 * 8 + 16_384)
    return "Use a file up to 64 MiB plus its header, with at most 8,388,608 values.";
  return "";
}

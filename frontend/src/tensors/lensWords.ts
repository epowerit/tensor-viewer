import type { Run } from "../api/client";
import { tokenize } from "../inputs/samples";

/**
 * The sentence the model's positions read: the text input whose length
 * matches them, with any what-if words set, and its vocabulary by id.
 */
export function lensWords(run: Run, positions: number) {
  const inputs = [
    { name: run.project.input_name ?? "x", input: run.project.input },
    ...(run.project.additional_inputs ?? []),
  ];
  for (const { input } of inputs) {
    if (!input.text) continue;
    const read = tokenize(input.text);
    if (read.tokens.length !== positions) continue;
    const tokens = [...read.tokens];
    for (const edit of input.edits ?? [])
      if (read.vocabulary[edit.value] !== undefined)
        tokens[edit.index % tokens.length] = read.vocabulary[edit.value];
    return { tokens, vocabulary: read.vocabulary };
  }
  return null;
}

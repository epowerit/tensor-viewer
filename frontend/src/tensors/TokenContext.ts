import { createContext, useContext } from "react";
import type { Run, Tensor } from "../api/client";
import { tokenize } from "../inputs/samples";
import { rootLabel, type AxisStory } from "./axisLineage";

/**
 * The word at each position of a tensor's axis, or null when it has none:
 * the sentence's words along its positions, or its vocabulary along ids.
 */
export type TokenAxes = (
  tensor: Tensor,
  axis: number,
) => { words: string[]; vocabulary: boolean } | null;

/**
 * Words on token axes. A model fed a sentence carries its words along every
 * axis that came from the sentence's positions, through projections, head
 * splits and attention: an attention map's rows and columns are the words
 * that attend and are attended to. An axis named for the vocabulary reads
 * as the sentence's vocabulary, id by id.
 */
export function tokenAxes(
  run: Run,
  lineage: ((tensorId: string) => AxisStory[]) | null,
): TokenAxes | null {
  const inputs = [
    { name: run.project.input_name ?? "x", input: run.project.input },
    ...(run.project.additional_inputs ?? []).map((item) => ({
      name: item.name,
      input: item.input,
    })),
  ];
  const sources: { label: string; tokens: string[] }[] = [];
  let vocabulary: string[] | null = null;
  for (const { name, input } of inputs) {
    if (!input.text) continue;
    const tensor = run.trace.input_ids
      .map((id) => run.trace.tensors[id])
      .find((found) => found?.name === name);
    if (!tensor?.shape.length) continue;
    const read = tokenize(input.text);
    // A what-if run may have set some token ids to other words.
    const tokens = [...read.tokens];
    for (const edit of input.edits ?? [])
      if (read.vocabulary[edit.value] !== undefined)
        tokens[edit.index % tokens.length] = read.vocabulary[edit.value];
    // Token positions run along the input's last axis.
    sources.push({
      label: rootLabel(tensor, tensor.shape.length - 1),
      tokens,
    });
    vocabulary ??= read.vocabulary;
  }
  if (!sources.length) return null;
  return (tensor, axis) => {
    const size = tensor.shape[axis];
    const story = lineage?.(tensor.id)?.[axis];
    const term = story?.terms.length === 1 ? story.terms[0] : undefined;
    if (term && !term.part) {
      const source = sources.find(
        (found) => found.label === term.label && found.tokens.length === size,
      );
      if (source) return { words: source.tokens, vocabulary: false };
    }
    if (
      vocabulary &&
      /vocab/i.test(tensor.axes[axis] ?? "") &&
      size >= vocabulary.length
    )
      return {
        words: Array.from(
          { length: size },
          (_, id) => vocabulary![id] ?? `#${id}`,
        ),
        vocabulary: true,
      };
    return null;
  };
}

export const TokenContext = createContext<TokenAxes | null>(null);

/** The words along a tensor's axes, where it has any. */
export function useTokenAxes(
  tensor: Tensor,
): (axis: number | null) => string[] | null {
  const tokens = useContext(TokenContext);
  return (axis) =>
    axis === null || !tokens ? null : (tokens(tensor, axis)?.words ?? null);
}

/**
 * A sentence's positions and a vocabulary axis on one tensor, as `logits` or
 * the probabilities after them have: what the model predicts at each word.
 */
export function usePredictionAxes(tensor: Tensor): {
  positions: number;
  vocabulary: number;
  words: string[];
  ids: string[];
} | null {
  const tokens = useContext(TokenContext);
  if (!tokens) return null;
  const found = tensor.shape.map((_, axis) => tokens(tensor, axis));
  const positions = found.findIndex((item) => item && !item.vocabulary);
  const vocabulary = found.findIndex((item) => item?.vocabulary);
  return positions >= 0 && vocabulary >= 0
    ? {
        positions,
        vocabulary,
        words: found[positions]!.words,
        ids: found[vocabulary]!.words,
      }
    : null;
}

/** A word short enough for a grid's margin: up to six characters. */
export const shortWord = (word: string) =>
  word.length > 6 ? `${word.slice(0, 5)}…` : word;

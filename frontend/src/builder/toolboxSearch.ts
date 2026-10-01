import type { ToolboxItem } from "../api/client";

// These aliases describe existing tools; they do not add general reshape,
// arbitrary permutations, or recurrent variants that the catalog cannot build.
const aliases: Record<string, string> = {
  conv1d: "CNN convolutional neural network convolution layer conv 1d",
  conv2d: "CNN convolutional neural network convolution layer conv 2d",
  linear: "linear layer dense fully connected affine",
  rnn: "RNN recurrent neural network recurrent layer",
  attention: "multi head multihead self attention selfattention MHA",
  split_axis: "reshape split dimension",
  merge_axes: "reshape merge dimensions",
};

function words(text: string, splitCase = true): string[] {
  const normalized = text.normalize("NFKC");
  return (
    (splitCase
      ? normalized
          .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
          .replace(/([a-z\d])([A-Z])/g, "$1 $2")
      : normalized
    )
      .toLowerCase()
      .match(/[\p{L}\p{N}]+/gu) ?? []
  );
}

function matches(text: string, queryWords: string[]): boolean {
  const searchable = [...new Set([...words(text), ...words(text, false)])];
  return queryWords.every((query) =>
    searchable.some((word) => word.startsWith(query)),
  );
}

/** Match every query word, in any order, without reordering the catalog. */
export function filterToolboxItems(
  catalog: ToolboxItem[],
  query: string,
): ToolboxItem[] {
  const queryWords = words(query);
  if (!queryWords.length) return catalog;
  return catalog.filter((item) =>
    matches(
      [
        item.kind,
        item.title,
        item.description,
        item.group,
        aliases[item.kind] ?? "",
        item.custom?.name ?? "",
        item.custom?.description ?? "",
        item.custom?.class_name ?? "",
      ].join(" "),
      queryWords,
    ),
  );
}

/** The input tool is separate from the component catalog and its categories. */
export function matchesInputTool(query: string): boolean {
  const queryWords = words(query);
  return (
    queryWords.length > 0 &&
    matches("Input tensor set dimensions shape and starting values", queryWords)
  );
}

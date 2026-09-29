import type { Draft, ForwardInput } from "../api/client";
import { inputIssue } from "./fixtures";

const PYTHON_KEYWORDS = new Set(
  "False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split(
    " ",
  ),
);

export function forwardInputs(draft: Draft): ForwardInput[] {
  return [
    {
      name: draft.input_name ?? "x",
      binding: draft.input_binding ?? "positional",
      input: draft.input,
    },
    ...(draft.additional_inputs ?? []),
  ];
}

export function withForwardInputs(draft: Draft, inputs: ForwardInput[]): Draft {
  const [first, ...additional_inputs] = inputs;
  return {
    ...draft,
    input: first.input,
    input_name: first.name,
    input_binding: first.binding,
    additional_inputs,
  };
}

export function forwardCall(inputs: ForwardInput[]): string {
  return `model(${inputs.map((item) => (item.binding === "keyword" ? `${item.name}=${item.name}` : item.name)).join(", ")})`;
}

export function forwardIssue(inputs: ForwardInput[], mode = "values"): string {
  if (!inputs.length || inputs.length > 8)
    return "Configure between one and eight tensor inputs.";
  const names = new Set<string>();
  let keywordSeen = false;
  for (const item of inputs) {
    if (
      !/^[A-Za-z_][A-Za-z0-9_]*$/.test(item.name) ||
      item.name.length > 100 ||
      PYTHON_KEYWORDS.has(item.name)
    )
      return "Use a Python identifier for each input name, such as query or attention_mask.";
    if (names.has(item.name))
      return `The input name “${item.name}” is already in use.`;
    names.add(item.name);
    if (keywordSeen && item.binding === "positional")
      return "Place positional inputs before keyword inputs.";
    keywordSeen ||= item.binding === "keyword";
    const issue = inputIssue(item.input, mode);
    if (issue) return `${item.name}: ${issue}`;
  }
  if (
    mode === "values" &&
    inputs.reduce(
      (sum, item) => sum + item.input.shape.reduce((a, b) => a * b, 1),
      0,
    ) > 32_000_000
  )
    return "Use Shapes only for more than 32 million total input elements.";
  return "";
}

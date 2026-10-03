import { createContext } from "react";
import type { LineResult } from "../console/script";
import type { ContractCheck } from "./contracts";

export type ContractIndex = {
  /** The checked contract on the line that produced each tensor state. */
  byTensor: Map<string, ContractCheck>;
  /** The same, by the step that produced it. */
  byOperation: Map<string, ContractCheck>;
  /** Rewrite a line's broken value contracts to what the run recorded. */
  accept?: (line: number) => void;
};

/**
 * Where checked contracts land: the tensor and step each line produced, so a
 * tensor card, the shelf, and the flow table can say whether it keeps them.
 */
export function contractIndex(
  marks: ReadonlyMap<number, ContractCheck>,
  results: ReadonlyMap<number, LineResult>,
): ContractIndex {
  const byTensor = new Map<string, ContractCheck>();
  const byOperation = new Map<string, ContractCheck>();
  marks.forEach((check, line) => {
    // An unreadable contract has no text; it is an editor problem only.
    if (!check.text) return;
    const result = results.get(line);
    if (!result?.fresh || result.predicted) return;
    if (result.output) byTensor.set(result.output.id, check);
    const step = result.operations.at(-1)?.id;
    if (step) byOperation.set(step, check);
  });
  return { byTensor, byOperation };
}

export const ContractContext = createContext<ContractIndex | null>(null);

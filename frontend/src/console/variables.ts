import type { Run, Tensor } from "../api/client";

export type Variable = {
  name: string;
  /** The latest recorded state carrying this name. */
  tensor: Tensor;
  /** Journey node that produced the latest state. */
  nodeId: string;
  line: number | null;
  /** How many recorded states have had this name. */
  states: number;
  /** Other current variables backed by the same storage. */
  sharedWith: string[];
  /** Named only after the operation that produced it. */
  anonymous: boolean;
};

/** Named tensors as they stand at the end of the run, in order of first use. */
export function variables(trace: Run["trace"]): Variable[] {
  const found = new Map<string, Variable>();
  const record = (
    id: string,
    nodeId: string,
    line: number | null,
    kind?: string,
  ) => {
    const tensor = trace.tensors[id];
    if (!tensor || tensor.role === "parameter") return;
    const previous = found.get(tensor.name);
    found.set(tensor.name, {
      name: tensor.name,
      tensor,
      nodeId,
      line,
      states: (previous?.states ?? 0) + 1,
      sharedWith: [],
      anonymous: tensor.name === kind,
    });
  };
  trace.input_ids.forEach((id) => record(id, `input-${id}`, null));
  for (const op of trace.operations) {
    const line = op.source && !op.source.file ? op.source.line : null;
    op.outputs.forEach((id) => record(id, op.id, line, op.kind));
    op.mutations?.forEach((mutation) => {
      // A write gives an existing name a new state without reassigning it.
      if (!op.outputs.includes(mutation.after))
        record(mutation.after, op.id, line);
    });
  }
  const list = [...found.values()];
  for (const item of list)
    item.sharedWith = list
      .filter(
        (other) =>
          other !== item && other.tensor.storage_id === item.tensor.storage_id,
      )
      .map((other) => other.name);
  return list;
}
